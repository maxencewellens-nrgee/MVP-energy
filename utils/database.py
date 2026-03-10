import streamlit as st
from supabase import create_client, Client
from typing import Optional, Dict, List, Any
import pandas as pd
from datetime import date

def get_supabase_client() -> Client:
    """Initialize and return Supabase client."""
    url = st.secrets.get("SUPABASE_URL", "")
    key = st.secrets.get("SUPABASE_ANON_KEY", "")

    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_ANON_KEY in secrets")

    return create_client(url, key)

class DatabaseManager:
    """Centralized database operations manager."""

    def __init__(self, client: Client):
        self.client = client

    def get_or_create_contract(self, user_id: str, year: int, defaults: Dict[str, Any]) -> Optional[Dict]:
        """Get existing contract or create new one with defaults."""
        try:
            response = self.client.table("contracts").select("*").eq("user_id", user_id).eq("year", year).maybeSingle().execute()

            if response.data:
                return response.data

            contract_data = {
                "user_id": user_id,
                "year": year,
                "total_mwh": defaults.get("total_mwh", 200.0),
                "max_fixations": defaults.get("max_fixations", 5),
                "dso": defaults.get("dso", "ORES"),
                "segment": defaults.get("segment", "BT")
            }

            response = self.client.table("contracts").insert(contract_data).execute()
            return response.data[0] if response.data else None

        except Exception as e:
            st.error(f"Error managing contract: {e}")
            return None

    def update_contract(self, contract_id: str, updates: Dict[str, Any]) -> bool:
        """Update contract with given fields."""
        try:
            self.client.table("contracts").update(updates).eq("id", contract_id).execute()
            return True
        except Exception as e:
            st.error(f"Error updating contract: {e}")
            return False

    def get_fixations(self, contract_id: str) -> List[Dict]:
        """Get all fixations for a contract."""
        try:
            response = self.client.table("fixations").select("*").eq("contract_id", contract_id).order("date", desc=True).execute()
            return response.data or []
        except Exception as e:
            st.error(f"Error fetching fixations: {e}")
            return []

    def add_fixation(self, contract_id: str, user_id: str, fixation_date: date, price: float, volume: float) -> bool:
        """Add a new fixation."""
        try:
            fixation_data = {
                "contract_id": contract_id,
                "user_id": user_id,
                "date": fixation_date.isoformat(),
                "price": float(price),
                "volume": float(volume)
            }
            self.client.table("fixations").insert(fixation_data).execute()
            return True
        except Exception as e:
            st.error(f"Error adding fixation: {e}")
            return False

    def delete_fixation(self, fixation_id: str) -> bool:
        """Delete a fixation."""
        try:
            self.client.table("fixations").delete().eq("id", fixation_id).execute()
            return True
        except Exception as e:
            st.error(f"Error deleting fixation: {e}")
            return False

    def get_past_contract(self, user_id: str, year: int) -> Optional[Dict]:
        """Get past contract data."""
        try:
            response = self.client.table("past_contracts").select("*").eq("user_id", user_id).eq("year", year).maybeSingle().execute()
            return response.data
        except Exception as e:
            st.error(f"Error fetching past contract: {e}")
            return None

    def upsert_past_contract(self, user_id: str, year: int, fixed_volume: float, fixed_price: float) -> bool:
        """Create or update past contract."""
        try:
            contract_data = {
                "user_id": user_id,
                "year": year,
                "fixed_volume": float(fixed_volume),
                "fixed_price": float(fixed_price)
            }
            self.client.table("past_contracts").upsert(contract_data, on_conflict="user_id,year").execute()
            return True
        except Exception as e:
            st.error(f"Error upserting past contract: {e}")
            return False

    def get_market_data_cache(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get cached market data."""
        try:
            response = self.client.table("market_data_cache").select("*").gte("date", start_date).lte("date", end_date).order("date").execute()

            if response.data:
                df = pd.DataFrame(response.data)
                df = df.rename(columns={
                    "avg_price": "avg",
                    "min_price": "mn",
                    "max_price": "mx",
                    "data_points": "n"
                })
                df["date"] = pd.to_datetime(df["date"]).dt.date
                return df[["date", "avg", "mn", "mx", "n"]]

            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error fetching market cache: {e}")
            return pd.DataFrame()

    def cache_market_data(self, df: pd.DataFrame) -> bool:
        """Cache market data."""
        try:
            records = []
            for _, row in df.iterrows():
                records.append({
                    "date": pd.to_datetime(row["date"]).date().isoformat(),
                    "avg_price": float(row["avg"]),
                    "min_price": float(row["mn"]),
                    "max_price": float(row["mx"]),
                    "data_points": int(row["n"])
                })

            if records:
                self.client.table("market_data_cache").upsert(records, on_conflict="date").execute()
            return True
        except Exception as e:
            st.error(f"Error caching market data: {e}")
            return False
