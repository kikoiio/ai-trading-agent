"""RAG retriever: converts market state into semantic queries and retrieves relevant KB entries."""

import logging
from src.kb.vectorstore import KBVectorStore

logger = logging.getLogger(__name__)


class KBRetriever:
    """Retrieves relevant Al Brooks knowledge base entries based on market context."""

    def __init__(self, vectorstore: KBVectorStore):
        self.store = vectorstore

    def build_market_queries(self, market_data: dict) -> list[str]:
        """Convert current market state into 2-3 semantic search queries.

        Args:
            market_data: Dict with per-asset market sections containing
                         indicators, prices, and trend information.

        Returns:
            List of natural language queries for KB search.
        """
        queries = []

        # Analyze each asset's data to build context-aware queries
        for asset, data in market_data.items():
            if not isinstance(data, dict):
                continue

            intraday = data.get("intraday_5m", {})
            longterm = data.get("long_term_4h", {})

            # Query 1: Trend structure
            trend_parts = [f"{asset} price action"]
            ema20_4h = longterm.get("ema20")
            ema50_4h = longterm.get("ema50")
            if ema20_4h and ema50_4h:
                if isinstance(ema20_4h, (int, float)) and isinstance(ema50_4h, (int, float)):
                    if ema20_4h > ema50_4h:
                        trend_parts.append("bull trend EMA20 above EMA50")
                    else:
                        trend_parts.append("bear trend EMA20 below EMA50")

            rsi = longterm.get("rsi14")
            if isinstance(rsi, (int, float)):
                if rsi > 70:
                    trend_parts.append("overbought RSI")
                elif rsi < 30:
                    trend_parts.append("oversold RSI")

            queries.append(" ".join(trend_parts))

            # Query 2: Pattern/momentum based
            macd_data = longterm.get("macd", {})
            if isinstance(macd_data, dict):
                macd_val = macd_data.get("valueMACD") or macd_data.get("value")
                if isinstance(macd_val, (int, float)):
                    if macd_val > 0:
                        queries.append(f"{asset} bullish momentum MACD positive entry setup")
                    else:
                        queries.append(f"{asset} bearish momentum MACD negative reversal pattern")

            # Query 3: Volatility/range context
            atr = longterm.get("atr14")
            if isinstance(atr, (int, float)):
                current_price = data.get("current_price")
                if isinstance(current_price, (int, float)) and current_price > 0:
                    atr_pct = (atr / current_price) * 100
                    if atr_pct > 3:
                        queries.append(f"{asset} high volatility wide range bars breakout")
                    elif atr_pct < 1:
                        queries.append(f"{asset} low volatility tight trading range compression")

        # Deduplicate and limit
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique[:3]

    def retrieve_for_context(self, market_data: dict, n_per_query: int = 3) -> list[dict]:
        """Pre-fetch relevant KB entries based on current market state.

        Args:
            market_data: Market data dict keyed by asset.
            n_per_query: Number of results per query.

        Returns:
            List of unique KB entry dicts with id, document, metadata.
        """
        if self.store.collection.count() == 0:
            return []

        queries = self.build_market_queries(market_data)
        if not queries:
            # Fallback: generic query
            queries = ["price action trading entry signal bar pattern"]

        results = []
        seen_ids = set()

        for query in queries:
            try:
                hits = self.store.query(query, n_results=n_per_query)
                for hit in hits:
                    if hit["id"] not in seen_ids:
                        results.append(hit)
                        seen_ids.add(hit["id"])
            except Exception as e:
                logger.error("KB retrieval failed for query '%s': %s", query, e)

        logger.info("Retrieved %d KB entries from %d queries", len(results), len(queries))
        return results

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Direct search for use by the LLM tool.

        Args:
            query: Natural language search query.
            n_results: Max results to return.

        Returns:
            List of KB entry dicts.
        """
        if self.store.collection.count() == 0:
            return []
        return self.store.query(query, n_results=n_results)

    def format_entries_for_prompt(self, entries: list[dict], max_chars_per_entry: int = 600) -> str:
        """Format retrieved KB entries for injection into the LLM prompt.

        Args:
            entries: List of entry dicts from query().
            max_chars_per_entry: Max chars per entry content.

        Returns:
            Formatted string for prompt injection.
        """
        if not entries:
            return ""

        parts = ["--- KNOWLEDGE BASE ENTRIES ---"]
        for entry in entries:
            entry_id = entry.get("id", "?")
            meta = entry.get("metadata", {})
            title = meta.get("slide_title", "Untitled")
            chapter = meta.get("chapter", "?")
            slide_num = meta.get("slide_number", "?")
            content = entry.get("document", "")[:max_chars_per_entry]
            if len(entry.get("document", "")) > max_chars_per_entry:
                content += "..."

            parts.append(f"[{entry_id}] \"{title}\"")
            parts.append(f"Source: {chapter}, Slide {slide_num}")
            parts.append(content)
            parts.append("---")

        parts.append("--- END KNOWLEDGE BASE ---")
        return "\n".join(parts)

    def get_available_ids(self, entries: list[dict]) -> list[str]:
        """Get list of entry IDs from retrieved entries."""
        return [e.get("id", "") for e in entries if e.get("id")]
