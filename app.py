import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import Optional
import numpy as np

# Page configuration
st.set_page_config(
    page_title="🛡️ Warranty Analysis Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }

    .warranty-highlight {
        background-color: #fef3c7;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }

    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def extract_warranty_excerpt(description: str) -> str:
    """Extract warranty-related excerpt from description with highlighting."""
    if not description or pd.isna(description):
        return ""

    warranty_terms = [
        "warranty",
        "warrantee",
        "waranty",
        "guarantee",
        "guaranteed",
        "guarante",
        "cover",
        "coverage",
        "protection",
        "protected",
        "assured",
        "assurance",
    ]

    text_lower = description.lower()

    # Find the first occurrence of any warranty term
    earliest_match = len(description)
    matched_term = ""

    for term in warranty_terms:
        index = text_lower.find(term)
        if index != -1 and index < earliest_match:
            earliest_match = index
            matched_term = term

    if earliest_match == len(description):
        return ""

    # Extract context around the warranty mention
    start = max(0, earliest_match - 60)
    end = min(len(description), earliest_match + len(matched_term) + 100)

    excerpt = description[start:end].strip()

    # Add ellipsis if truncated
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(description):
        excerpt = excerpt + "..."

    return excerpt


def highlight_warranty_terms(text: str) -> str:
    """Highlight warranty terms in text for display."""
    if not text:
        return ""

    warranty_terms = [
        "warranty",
        "warrantee",
        "waranty",
        "guarantee",
        "guaranteed",
        "guarante",
        "cover",
        "coverage",
        "protection",
        "protected",
        "assured",
        "assurance",
    ]

    pattern = r"\b(" + "|".join(warranty_terms) + r")\b"
    highlighted = re.sub(
        pattern,
        r'<span class="warranty-highlight">\1</span>',
        text,
        flags=re.IGNORECASE,
    )

    return highlighted


def load_and_process_data(uploaded_file) -> pd.DataFrame:
    """Load and process the warranty CSV data."""
    try:
        df = pd.read_csv(uploaded_file)

        # Ensure required columns exist
        required_columns = ["id", "dhf_dealer_id", "title", "price", "full_description"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None

        # Clean and process data
        df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
        df["warranty_excerpt"] = df["full_description"].apply(extract_warranty_excerpt)

        # Create warranty period display
        if "warranty_period_type" in df.columns:
            df["warranty_period_display"] = (
                df["warranty_period_type"]
                .fillna("No Period")
                .str.replace("_", " ")
                .str.title()
            )
        else:
            df["warranty_period_display"] = "No Period"

        return df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def create_summary_metrics(df: pd.DataFrame):
    """Create summary metrics cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="🚗 Total Warranty Listings", value=f"{len(df):,}")

    with col2:
        unique_dealers = df["dhf_dealer_id"].nunique()
        st.metric(label="🏢 Unique Dealers", value=f"{unique_dealers:,}")

    with col3:
        if "warranty_period_type" in df.columns:
            with_periods = df["warranty_period_type"].notna().sum()
        else:
            with_periods = 0
        st.metric(label="⏰ With Specific Periods", value=f"{with_periods:,}")

    with col4:
        avg_price = df["price"].mean()
        st.metric(label="💰 Average Price", value=f"€{avg_price:,.0f}")


def create_charts(df: pd.DataFrame):
    """Create visualization charts."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Warranty Periods Distribution")
        if "warranty_period_type" in df.columns:
            period_counts = df["warranty_period_display"].value_counts()
            fig_pie = px.pie(
                values=period_counts.values,
                names=period_counts.index,
                title="Distribution of Warranty Periods",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No warranty period data available")

    with col2:
        st.subheader("🏆 Top Dealers by Warranty Count")
        dealer_counts = df["dhf_dealer_id"].value_counts().head(10)
        dealer_names = [f"Dealer {dealer_id}" for dealer_id in dealer_counts.index]

        fig_bar = px.bar(
            x=dealer_names,
            y=dealer_counts.values,
            title="Top 10 Dealers by Warranty Listings",
            labels={"x": "Dealer", "y": "Number of Listings"},
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters to the dataframe."""
    st.sidebar.header("🔍 Filters")

    # Search filter
    search_term = st.sidebar.text_input(
        "🔎 Search Titles", placeholder="Enter search term..."
    )
    if search_term:
        df = df[df["title"].str.contains(search_term, case=False, na=False)]

    # Warranty period filter
    if "warranty_period_type" in df.columns:
        periods = ["All"] + sorted(df["warranty_period_display"].unique().tolist())
        selected_period = st.sidebar.selectbox("⏰ Warranty Period", periods)
        if selected_period != "All":
            df = df[df["warranty_period_display"] == selected_period]

    # Dealer filter
    dealers = ["All"] + sorted([f"Dealer {d}" for d in df["dhf_dealer_id"].unique()])
    selected_dealer = st.sidebar.selectbox("🏢 Dealer", dealers)
    if selected_dealer != "All":
        dealer_id = int(selected_dealer.replace("Dealer ", ""))
        df = df[df["dhf_dealer_id"] == dealer_id]

    # Price range filter
    st.sidebar.subheader("💰 Price Range")
    min_price = st.sidebar.number_input("Min Price (€)", min_value=0, value=0)
    max_price = st.sidebar.number_input(
        "Max Price (€)",
        min_value=0,
        value=int(df["price"].max()) if not df.empty else 100000,
    )

    df = df[(df["price"] >= min_price) & (df["price"] <= max_price)]

    # Show filter summary
    st.sidebar.markdown(f"**Filtered Results:** {len(df):,} records")

    return df


def display_data_table(df: pd.DataFrame):
    """Display the filtered data in a table."""
    st.subheader("📋 Warranty Listings")

    if df.empty:
        st.warning("No data matches the current filters.")
        return

    # Prepare display dataframe
    display_df = df.copy()

    # Format columns for display
    display_df["Price"] = display_df["price"].apply(lambda x: f"€{x:,.0f}")
    display_df["Dealer"] = display_df["dhf_dealer_id"].apply(lambda x: f"Dealer {x}")

    # Select columns to display
    columns_to_show = ["id", "Dealer", "title", "Price"]

    if "warranty_period_type" in df.columns:
        columns_to_show.append("warranty_period_display")

    if "warranty_keywords" in df.columns:
        columns_to_show.append("warranty_keywords")

    columns_to_show.extend(["warranty_excerpt", "full_description"])

    # Rename columns for better display
    column_mapping = {
        "id": "ID",
        "title": "Title",
        "warranty_period_display": "Warranty Period",
        "warranty_keywords": "Keywords",
        "warranty_excerpt": "Warranty Excerpt",
        "full_description": "Full Description",
    }

    display_columns = [col for col in columns_to_show if col in display_df.columns]
    table_df = display_df[display_columns].rename(columns=column_mapping)

    # Limit rows for performance
    max_rows = 100
    if len(table_df) > max_rows:
        st.info(f"Showing first {max_rows} of {len(table_df):,} results")
        table_df = table_df.head(max_rows)

    # Display table
    st.dataframe(table_df, use_container_width=True, height=400)


def export_data(df: pd.DataFrame):
    """Provide data export functionality."""
    st.subheader("📥 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Export Filtered Data as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"warranty_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    with col2:
        if st.button("📋 Export Summary Report"):
            summary = f"""
# Warranty Analysis Summary Report
Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
- Total Listings: {len(df):,}
- Unique Dealers: {df["dhf_dealer_id"].nunique():,}
- Average Price: €{df["price"].mean():,.0f}
- Price Range: €{df["price"].min():,.0f} - €{df["price"].max():,.0f}

## Warranty Periods
{df["warranty_period_display"].value_counts().to_string() if "warranty_period_display" in df.columns else "No warranty period data"}

## Top Dealers
{df["dhf_dealer_id"].value_counts().head(10).to_string()}
            """

            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"warranty_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )


def main():
    """Main application function."""
    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🛡️ Warranty Analysis Dashboard</h1>
        <p>Interactive analysis of warranty mentions across vehicle listings</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload your warranty CSV file",
        type=["csv"],
        help="Upload the CSV file exported from your warranty_ads database view",
    )

    if uploaded_file is not None:
        # Load and process data
        with st.spinner("Loading and processing data..."):
            df = load_and_process_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Successfully loaded {len(df):,} warranty listings!")

            # Apply filters
            filtered_df = apply_filters(df)

            # Display summary metrics
            create_summary_metrics(filtered_df)

            st.markdown("---")

            # Create visualizations
            create_charts(filtered_df)

            st.markdown("---")

            # Display data table
            display_data_table(filtered_df)

            st.markdown("---")

            # Export functionality
            export_data(filtered_df)

            # Additional insights
            with st.expander("🔍 Additional Insights"):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Price Statistics")
                    st.write(filtered_df["price"].describe())

                with col2:
                    st.subheader("Data Quality")
                    st.write(
                        f"Records with excerpts: {(filtered_df['warranty_excerpt'] != '').sum():,}"
                    )
                    st.write(
                        f"Records with full descriptions: {filtered_df['full_description'].notna().sum():,}"
                    )

    else:
        st.info("👆 Please upload your warranty CSV file to get started")

        st.markdown("""
        ### Expected CSV Columns:
        - `id` - Listing ID
        - `dhf_dealer_id` - Dealer ID
        - `title` - Vehicle title
        - `price` - Price in euros
        - `full_description` - Complete description text
        - `warranty_period_type` - Warranty period (optional)
        - `warranty_keywords` - Warranty keywords (optional)
        """)


if __name__ == "__main__":
    main()
