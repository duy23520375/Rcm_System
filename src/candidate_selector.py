from itertools import chain
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple

import polars as pl
from polars._typing import EngineType
from tqdm import tqdm

from src.reporter import Reporter


class CandidateSelector:
    """
    Multi-strategy candidate selection for e-commerce recommender system.

    Architecture:
    1. Multiple candidate sources generate ~200 candidates total
    2. ML/DL model scores these candidates
    3. Final ranking produces top-K recommendations

    For purchase prediction, we combine:
    - Collaborative filtering (what similar users bought)
    - Content-based (similar items to user's history)
    - Business rules (trending, seasonal, promotions)
    """

    def __init__(
        self,
        purchase_history_lf: pl.LazyFrame,
        user_lf: pl.LazyFrame,
        item_lf: pl.LazyFrame,
        engine: EngineType = "streaming",
    ):  # [item_id, brand, category, price, ...]

        self.purchase_history_lf = purchase_history_lf
        self.user_lf = user_lf
        self.item_lf = item_lf

        self.engine = engine
        self.reporter = Reporter()

        # Precompute for efficiency
        self._build_lookup_structures()

    def _build_lookup_structures(self):
        """Build efficient lookup structures"""
        # User purchase history - collect and cache as dict for O(1) lookup
        self.reporter.start_timing("Build lookup structures")
        purchase_history_df = (
            self.purchase_history_lf
            .sort(["customer_id", "created_date"])
            .collect(engine=self.engine)
        )

        user_purchased_df = (
            purchase_history_df
            .select(["customer_id", "item_id"])
            .group_by("customer_id")
            .agg(pl.col("item_id").unique().alias("items"))
        )

        self.user_purchased_items_dict: Dict[int, List[str]] = dict(zip(
            user_purchased_df["customer_id"].to_list(),
            user_purchased_df["items"].to_list(),
        ))
        self.reporter.end_timing()

        # Item co-purchase matrix (items bought together)
        self.reporter.start_timing("Build item co-purchase matrix")
        self.item_copurchase = self._build_copurchase_matrix(purchase_history_df)
        self.reporter.end_timing()

        # Item popularity - collect once and cache as DataFrame
        self.reporter.start_timing("Compute item popularity")
        self.item_popularity_df = self._compute_item_popularity()
        # Create dict for fast lookup
        self.item_popularity_dict: Dict[str, float] = {
            row["item_id"]: row["popularity_score"]
            for row in self.item_popularity_df.iter_rows(named=True)
        }
        # Cache top popular items (for fallback)
        self.top_popular_items = (
            self.item_popularity_df.sort("popularity_score", descending=True)
            .head(200)["item_id"]
            .to_list()
        )
        self.reporter.end_timing()

        # Item-item similarity (based on features)
        self.reporter.start_timing("Build item similarity map")
        self.item_similarity_map = self._build_item_similarity()
        self.reporter.end_timing()

        # Pre-collect item features for content-based filtering
        self.reporter.start_timing("Collect item features")
        self.item_features_df = self.item_lf.select(
            ["item_id", "brand", "category", "price"]
        ).collect(engine=self.engine)
        # Create lookup dict for item features
        self.item_features_dict: Dict[str, Tuple] = {
            row["item_id"]: (row["brand"], row["category"], row["price"])
            for row in self.item_features_df.iter_rows(named=True)
        }
        # Group items by brand and category for fast lookup
        self.items_by_brand: Dict[str, List[str]] = {}
        self.items_by_category: Dict[str, List[str]] = {}
        for row in self.item_features_df.iter_rows(named=True):
            item_id = row["item_id"]
            brand = row["brand"]
            category = row["category"]
            if brand not in self.items_by_brand:
                self.items_by_brand[brand] = []
            self.items_by_brand[brand].append(item_id)
            if category not in self.items_by_category:
                self.items_by_category[category] = []
            self.items_by_category[category].append(item_id)
        self.reporter.end_timing()

        # Pre-collect user purchases with features for content-based filtering
        self.reporter.start_timing("Collect user purchases with features")
        self.user_purchases_df = (
            self.purchase_history_lf.select(["customer_id", "item_id"])
            .join(
                self.item_lf.select(["item_id", "brand", "category", "price"]),
                on="item_id",
            )
            .collect(engine=self.engine)
        )
        self.reporter.end_timing()

        # Pre-cache business rule candidates (not user-specific)
        self.reporter.start_timing("Cache business rule candidates")
        self._cached_business_candidates = self._compute_business_rule_candidates()
        self.reporter.end_timing()

        # Pre-cache user reorder candidates
        self.reporter.start_timing("Cache user reorder candidates")
        self._user_reorder_candidates = self._build_user_reorder_candidates()
        self.reporter.end_timing()

    def _build_copurchase_matrix(
        self, purchase_history_df: pl.DataFrame
    ) -> Dict[str, List[str]]:
        """Build item co-purchase matrix: items frequently bought together"""

        transitions = (
            purchase_history_df
            .sort(["customer_id", "created_date"])
            .with_columns([
                # Get next item in sequence per customer
                pl.col("item_id").shift(-1).over("customer_id").alias("next_item")
            ])
            # Filter out last items (no transition)
            .filter(pl.col("next_item").is_not_null())
            # Group and count transitions
            .group_by(["item_id", "next_item"])
            .agg(pl.len().alias("count"))
            # Rank within each source item
            .with_columns([
                pl.col("count")
                .rank(method="ordinal", descending=True)
                .over("item_id")
                .alias("rank")
            ])
            # Keep only top 50 per source item
            .filter(pl.col("rank") <= 50)
            .sort(["item_id", "rank"])
            .group_by("item_id")
            .agg(
                pl.col("next_item").alias("items"),
            )
        )

        return dict(zip(transitions["item_id"].to_list(), transitions["items"].to_list()))


    def _compute_item_popularity(self) -> pl.DataFrame:
        """Compute item popularity metrics - returns collected DataFrame"""
        popularity = (
            self.purchase_history_lf.group_by("item_id")
            .agg(
                purchase_count=pl.len(),
                total_quantity=pl.sum("quantity"),
                unique_buyers=pl.n_unique("customer_id"),
            )
            .join(self.item_lf.select(["item_id", "category"]), on="item_id")
            .with_columns(
                [
                    # Popularity score combining recency and frequency
                    (pl.col("purchase_count") * pl.col("unique_buyers"))
                    .sqrt()
                    .alias("popularity_score")
                ]
            )
            .collect(engine=self.engine)
        )
        return popularity

    def _build_item_similarity(self) -> Dict[str, List[str]]:
        """Build item-item similarity based on features (brand, category, price range)"""
        # Collect item data once
        items_df = self.item_lf.select(["item_id", "category", "brand"]).collect(
            engine=self.engine
        )

        similarity_map = {}

        # Group items by category and brand for efficient lookup
        category_items = items_df.group_by("category").agg(
            pl.col("item_id").alias("category_items")
        )

        brand_items = items_df.group_by("brand").agg(
            pl.col("item_id").alias("brand_items")
        )

        # Create lookup dictionaries
        category_lookup = {
            row["category"]: row["category_items"]
            for row in category_items.iter_rows(named=True)
        }
        brand_lookup = {
            row["brand"]: row["brand_items"]
            for row in brand_items.iter_rows(named=True)
        }

        # Build similarity map efficiently
        for row in items_df.iter_rows(named=True):
            item_id = row["item_id"]
            category = row["category"]
            brand = row["brand"]

            # Get similar items (same category or brand)
            similar_set = set()

            # Add items from same category
            if category in category_lookup:
                similar_set.update(category_lookup[category])

            # Add items from same brand
            if brand in brand_lookup:
                similar_set.update(brand_lookup[brand])

            # Remove self
            similar_set.discard(item_id)

            # Convert to list and limit to top 30
            similarity_map[item_id] = list(similar_set)[:30]

        return similarity_map

    def _compute_business_rule_candidates(self) -> Set[str]:
        """Pre-compute business rule candidates (not user-specific)"""
        candidates = set()

        # Get items on promotion
        promoted_items = (
            self.purchase_history_lf.filter(pl.col("discount") > 0)
            .select("item_id")
            .unique()
            .head(10)
            .collect(engine=self.engine)
            .to_series()
            .to_list()
        )
        candidates.update(promoted_items)

        # New arrivals
        recent_date = datetime.now() - timedelta(days=30)
        new_arrivals = (
            self.item_lf.filter(pl.col("created_date") > pl.lit(recent_date))
            .select("item_id")
            .head(10)
            .collect(engine=self.engine)
            .to_series()
            .to_list()
        )
        candidates.update(new_arrivals)

        # High-margin items
        high_margin = (
            self.item_lf.sort("gp", descending=True)
            .select("item_id")
            .head(20)
            .collect(engine=self.engine)
            .to_series()
            .to_list()
        )
        candidates.update(high_margin)

        return candidates

    def _build_user_reorder_candidates(self) -> Dict[int, List[str]]:
        """Pre-compute reorder candidates for each user.
        Returns dict: customer_id -> list of item_ids (frequently purchased)
        """
        # Get purchase counts per user-item pair
        self.reporter.start_timing("Build user reorder candidates")
        user_item_counts = (
            self.purchase_history_lf.group_by(["customer_id", "item_id"])
            .agg(pl.len().alias("purchase_count"))
            .filter(pl.col("purchase_count") >= 2)  # Only items bought 2+ times
            .sort(["customer_id", "purchase_count"], descending=[False, True])
            .collect(engine=self.engine)
        )
        self.reporter.end_timing()

        result: Dict[int, List[str]] = {}

        for row in tqdm(user_item_counts.iter_rows(named=True)):
            customer_id = row["customer_id"]
            item_id = row["item_id"]

            if customer_id not in result:
                result[customer_id] = []

            if len(result[customer_id]) < 30:  # Keep top 30 per user
                result[customer_id].append(item_id)

        return result

    # ========================================================================
    # CANDIDATE GENERATION STRATEGIES
    # ========================================================================

    def get_collaborative_candidates(
        self, customer_id: int, n_candidates: int = 50
    ) -> Set[str]:
        """
        Strategy 1: Collaborative Filtering

        Logic: "Users who bought similar items also bought these"

        Methods:
        a) Item-based CF: Find items similar to user's purchase history
        b) Co-purchase: Items frequently bought together with user's items

        Why this works for e-commerce:
        - Captures buying patterns (people who buy X often buy Y)
        - Works well for repeat purchases and complements
        - Example: Buy phone → recommend phone case
        """
        candidates = set()

        # Get user's purchase history from cached dict (O(1) lookup)
        if customer_id not in self.user_purchased_items_dict:
            return candidates

        user_items = set(self.user_purchased_items_dict[customer_id])

        # Method A: Co-purchase (items bought together)
        copurchased = list(chain.from_iterable(
            self.item_copurchase.get(item[:10], []) for item in list(user_items)[:10]
        ))

        candidates.update(copurchased)

        # Method B: Similar items (item-item similarity)
        copurchased = list(chain.from_iterable(
            self.item_similarity_map.get(item[:5], []) for item in list(user_items)[:10]
        ))

        candidates.update(copurchased)

        # Remove items user already bought
        candidates -= user_items

        # Return top N by popularity using cached dict (no .collect())
        if len(candidates) > n_candidates:
            # Sort by popularity using cached dict
            sorted_candidates = sorted(
                candidates,
                key=lambda x: self.item_popularity_dict.get(x, 0),
                reverse=True,
            )
            return set(sorted_candidates[:n_candidates])

        return candidates

    def get_content_based_candidates(
        self, customer_id: int, n_candidates: int = 30
    ) -> Set[str]:
        """
        Strategy 2: Content-Based Filtering

        Logic: "Based on your purchase history, you might like these similar items"

        Uses:
        - Preferred brands (user bought Nike 5x → recommend Nike)
        - Preferred categories (user buys electronics → recommend electronics)
        - Price range affinity (user buys $50-100 items)

        Why this works for e-commerce:
        - Users have brand loyalty
        - Purchase patterns are category-specific
        - Price sensitivity matters
        """
        candidates = set()

        # Get user's purchase history with features
        user_purchases = self.user_purchases_df.filter(
            pl.col("customer_id") == customer_id
        )

        if len(user_purchases) == 0:
            return candidates

        # Find user's preferred brands
        brand_counts = (
            user_purchases.group_by("brand")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(3)
        )
        preferred_brands = brand_counts["brand"].to_list()

        # Find user's preferred categories
        category_counts = (
            user_purchases.group_by("category")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(3)
        )
        preferred_categories = category_counts["category"].to_list()

        # Find user's price range
        avg_price = user_purchases["price"].mean() or 0.0
        price_std = (
            user_purchases["price"].std()
            if len(user_purchases) > 1
            else avg_price * 0.3
        )
        price_min = avg_price - price_std
        price_max = avg_price + price_std

        # Get items matching user preferences
        user_purchased = set(user_purchases["item_id"].to_list())

        matching_items = (
            self.item_features_df.filter(
                (pl.col("brand").is_in(preferred_brands))
                | (pl.col("category").is_in(preferred_categories))
            )
            .filter(pl.col("price").is_between(price_min, price_max))
            .filter(~pl.col("item_id").is_in(user_purchased))
            .join(self.item_popularity_df, on="item_id")
            .sort("popularity_score", descending=True)
            .head(n_candidates)
        )

        candidates.update(matching_items["item_id"].to_list())
        return candidates

    def get_popularity_candidates(
        self, n_candidates: int = 30, filter_category: str | None = None
    ) -> Set[str]:
        """
        Strategy 3: Popularity-Based

        Logic: "Trending items that many people are buying"

        Use cases:
        - New users (cold start)
        - Seasonal items (winter coats in December)
        - Viral products

        Why this works for e-commerce:
        - Social proof (if many buy it, it's probably good)
        - Covers broad appeal items
        - Good fallback strategy
        """
        # Use pre-cached top popular items (no .collect() needed)
        if filter_category:
            # Filter by category using cached data
            filtered = [
                item_id
                for item_id in self.top_popular_items
                if item_id in self.item_features_dict
                and self.item_features_dict[item_id][1]
                == filter_category  # [1] is category
            ]
            return set(filtered[:n_candidates])

        return set(self.top_popular_items[:n_candidates])

    def get_business_rule_candidates(self, n_candidates: int = 20) -> Set[str]:
        """
        Strategy 4: Business Rules

        Logic: "Strategic items we want to promote"

        Examples:
        - Promoted items (sales, clearance)
        - High-margin products
        - New arrivals
        - Seasonal items
        - Personalized deals

        Why this works for e-commerce:
        - Aligns with business objectives
        - Promotes inventory movement
        - Increases profitability
        """
        # Return pre-cached business candidates (no .collect() needed)
        return set(list(self._cached_business_candidates)[:n_candidates])

    def get_reorder_candidates(
        self, customer_id: int, n_candidates: int = 20
    ) -> Set[str]:
        """
        Strategy 5: Reorder (Repurchase) Candidates

        Logic: "Items you've bought before and might need again"

        Use cases:
        - Consumables (groceries, toiletries)
        - Recurring purchases
        - Subscription-like behavior

        Why this works for e-commerce:
        - Many products are consumable
        - Customers have repeat purchase patterns
        - High conversion rate for reorders
        """
        # Use pre-cached user reorder candidates (no .collect() needed)
        if customer_id in self._user_reorder_candidates:
            return set(list(self._user_reorder_candidates[customer_id])[:n_candidates])

        # Fallback: return user's purchased items from cached dict
        if customer_id in self.user_purchased_items_dict:
            return set(self.user_purchased_items_dict[customer_id][:n_candidates])

        return set()

    # ========================================================================
    # UNIFIED CANDIDATE SELECTION
    # ========================================================================

    def get_candidates(
        self, customer_id: int, strategies_time: List[int] = [0.0] * 5, total_candidates: int = 200
    ) -> List[str]:
        """
        Unified candidate selection combining all strategies

        Allocation strategy (adjust based on your business):
        - 40% Collaborative (behavioral patterns)
        - 25% Content-based (user preferences)
        - 15% Reorder (repeat purchases)
        - 10% Popularity (trending items)
        - 10% Business rules (promotions, strategic)

        Returns:
            List of candidate item IDs for ML model scoring
        """
        candidates = set()

        # Strategy 1: Collaborative filtering (40% = 80 items)
        begin = time.time()
        collab_candidates = self.get_collaborative_candidates(
            customer_id, n_candidates=int(total_candidates * 0.4)
        )
        candidates.update(collab_candidates)
        strategies_time[0] += time.time() - begin

        # Strategy 2: Content-based (25% = 50 items)
        begin = time.time()
        content_candidates = self.get_content_based_candidates(
            customer_id, n_candidates=int(total_candidates * 0.25)
        )
        candidates.update(content_candidates)
        strategies_time[1] += time.time() - begin

        # Strategy 3: Reorder candidates (15% = 30 items)
        begin = time.time()
        reorder_candidates = self.get_reorder_candidates(
            customer_id, n_candidates=int(total_candidates * 0.15)
        )
        candidates.update(reorder_candidates)
        strategies_time[2] += time.time() - begin

        # Strategy 5: Business rules (10% = 20 items)
        begin = time.time()
        business_candidates = self.get_business_rule_candidates(
            n_candidates=int(total_candidates * 0.1)
        )
        candidates.update(business_candidates)
        strategies_time[3] += time.time() - begin

        # Strategy 4: Popularity (10% = 20 items)
        begin = time.time()
        popular_candidates = self.get_popularity_candidates(
            n_candidates=total_candidates - len(candidates)
        )
        candidates.update(popular_candidates)
        strategies_time[4] += time.time() - begin

        return list(candidates)
