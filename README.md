# 🛡️ Warranty Analysis Dashboard

Interactive Streamlit dashboard for analyzing warranty mentions across vehicle listings.

## Features

- 📊 **Interactive Analytics**: Real-time filtering and visualization
- 🔍 **Smart Warranty Extraction**: Automatically extracts warranty-related text from descriptions
- 📈 **Visual Charts**: Warranty period distribution and dealer analysis
- 📋 **Data Table**: Searchable and filterable listing table
- 📥 **Export Capabilities**: Download filtered data and summary reports
- ⚡ **High Performance**: Handles 18k+ records efficiently

## Requirements

- Python 3.8+
- Poetry for dependency management

## Installation

1. **Clone or create the project directory:**
```bash
mkdir warranty-analysis-dashboard
cd warranty-analysis-dashboard
```

2. **Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Initialize the project:**
```bash
poetry init
```

4. **Install dependencies:**
```bash
poetry install
```

## Usage

### Running the Dashboard

1. **Activate the virtual environment:**
```bash
poetry shell
```

2. **Run the Streamlit app:**
```bash
streamlit run app.py
```

3. **Open your browser** to the displayed URL (usually http://localhost:8501)

### Data Format

Your CSV file should contain these columns:
- `id` - Listing ID
- `dhf_dealer_id` - Dealer ID  
- `title` - Vehicle title
- `price` - Price in euros
- `full_description` - Complete description text
- `warranty_period_type` - Warranty period (optional)
- `warranty_keywords` - Warranty keywords (optional)

### Exporting Data from PostgreSQL

Use this query to export your warranty data:

```sql
COPY (
    SELECT * FROM warranty_ads 
    ORDER BY created_at DESC
) TO '/path/to/warranty_data.csv' 
WITH CSV HEADER;
```

## Dashboard Features

### 📊 Summary Metrics
- Total warranty listings
- Unique dealers count
- Listings with specific warranty periods
- Average price

### 🔍 Interactive Filters
- **Search**: Filter by vehicle title
- **Warranty Period**: Filter by specific warranty periods
- **Dealer**: Filter by specific dealers
- **Price Range**: Set minimum and maximum price filters

### 📈 Visualizations
- **Warranty Periods Distribution**: Pie chart showing breakdown of warranty periods
- **Top Dealers**: Bar chart of dealers with most warranty listings

### 📋 Data Table
- Smart warranty excerpt extraction with highlighting
- Full description column for detailed review
- Pagination for large datasets (shows first 100 results)

### 📥 Export Options
- **CSV Export**: Download filtered data as CSV
- **Summary Report**: Download text summary with key statistics

## Development

### Adding Dependencies
```bash
poetry add package-name
```

### Development Dependencies
```bash
poetry add --group dev package-name
```

### Code Formatting
```bash
poetry run black .
```

### Type Checking
```bash
poetry run mypy .
```

### Running Tests
```bash
poetry run pytest
```

## Deployment

### Streamlit Community Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click

### Local Network Access

To allow access from other devices on your network:
```bash
streamlit run app.py --server.address 0.0.0.0
```

## Configuration

### Environment Variables

You can set these environment variables:
- `STREAMLIT_SERVER_PORT` - Port number (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Server address (default: localhost)

### Streamlit Config

Create `.streamlit/config.toml` for additional configuration:
```toml
[server]
port = 8501
address = "0.0.0.0"

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## Troubleshooting

### Common Issues

1. **Large CSV files**: The dashboard shows first 100 results for performance. Use filters to narrow down results.

2. **Missing columns**: Ensure your CSV has the required columns listed above.

3. **Memory issues**: For very large datasets (50k+ records), consider:
   - Filtering data at the SQL level before export
   - Using chunked processing
   - Upgrading server memory

### Performance Tips

- Use specific filters to reduce dataset size
- Export smaller date ranges for analysis
- Consider creating summary tables for large historical datasets

## License

MIT License - feel free to modify and distribute.

## Support

For issues and feature requests, please create an issue in the repository.
