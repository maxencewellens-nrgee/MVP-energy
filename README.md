# MVP Énergie — Energy Contract Management

A modern energy contract management application for Belgian electricity markets with real-time market data, contract tracking, and cost analysis.

## Features

- **Market Data Visualization**: Real-time day-ahead electricity prices from ENTSO-E
- **Contract Management**: Track and manage energy fixations for 2024-2028
- **Cost Analysis**: Comprehensive cost breakdown including energy and network costs
- **User Authentication**: Secure login and data persistence with Supabase
- **Multi-year Planning**: Support for contracts across multiple years

## Architecture

### Modular Structure

```
project/
├── app.py                  # Main application
├── utils/
│   ├── database.py        # Supabase integration & data persistence
│   ├── formatters.py      # Number and currency formatting
│   ├── market_data.py     # ENTSO-E API & FlexyPower scraping
│   └── network_costs.py   # Belgian network cost tables
├── requirements.txt       # Python dependencies
└── .env.example          # Environment variables template
```

### Database Schema

**contracts**: User energy contracts with volume and settings
- Tracks total MWh, max fixations, DSO, and segment per year

**fixations**: Individual energy fixations (purchases)
- Date, price, and volume of each energy purchase

**past_contracts**: Historical contracts (2024-2025)
- Fixed volume and price for completed years

**market_data_cache**: Cached ENTSO-E market data
- Reduces API calls and improves performance

## Setup

### 1. Clone and Install

```bash
pip install -r requirements.txt
```

### 2. Configure Secrets

Create `.streamlit/secrets.toml`:

```toml
ENTSOE_TOKEN = "your_entso_e_token"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your_anon_key"
```

### 3. Database Setup

The database schema is automatically created via Supabase migrations. All tables have:
- Row Level Security (RLS) enabled
- User-specific access policies
- Proper indexes for performance
- Audit timestamps

### 4. Run Application

```bash
streamlit run app.py
```

## Usage

### Authentication

1. Create an account or log in
2. All data is private and user-specific

### Market Data

View historical day-ahead prices with:
- Interactive charts with moving averages
- Price statistics and summaries
- CAL (Calendar Year) forward prices

### Contract Management

1. **Past Contracts (2024-2025)**: Enter historical data
2. **Future Contracts (2026-2028)**:
   - Configure total volume and max fixations
   - Add individual fixations with date, price, volume
   - Track coverage percentage
3. **Cost Analysis**: View total costs including energy, network, and VAT

## Key Improvements

### Data Persistence
- All data stored in Supabase
- No data loss on refresh
- Multi-user support with RLS

### Error Handling
- Comprehensive try-catch blocks
- User-friendly error messages
- Graceful API failure handling

### Performance
- Market data caching
- Efficient database queries
- Optimized session state management

### Code Quality
- Modular architecture
- Type hints and validation
- DRY principles
- Clear separation of concerns

## Security

- Password authentication via Supabase Auth
- Row Level Security on all tables
- Input validation and sanitization
- No sensitive data in client code

## Network Costs

Supports Belgian DSOs:
- ORES
- RESA
- AIEG
- AIESH
- REW

Both BT (≤56 kVA) and MT (>56 kVA) segments for years 2026-2028.

## License

Proprietary
