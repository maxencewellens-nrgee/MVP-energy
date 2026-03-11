# Pre-Launch Checklist

## Required Configuration

### 1. ENTSOE API Token
- [ ] Create account at https://transparency.entsoe.eu/
- [ ] Request API token
- [ ] Add to `.streamlit/secrets.toml` as `ENTSOE_TOKEN`

### 2. Supabase Secrets
- [ ] Supabase project already created
- [ ] `SUPABASE_URL` available in `.streamlit/secrets.toml`
- [ ] `SUPABASE_ANON_KEY` available in `.streamlit/secrets.toml`

### 3. Environment Setup
Create `.streamlit/secrets.toml` with:
```toml
ENTSOE_TOKEN = "your_token_here"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your_anon_key"
```

## Database Status

### Tables Created ✓
- [x] contracts
- [x] fixations
- [x] past_contracts
- [x] market_data_cache

### RLS Enabled ✓
- [x] All tables have Row Level Security enabled
- [x] All policies configured
- [x] User isolation enforced

### Indexes ✓
- [x] contracts(user_id, year)
- [x] fixations(contract_id)
- [x] fixations(user_id)
- [x] past_contracts(user_id, year)
- [x] market_data_cache(date)

### Triggers ✓
- [x] Updated_at triggers on contracts
- [x] Updated_at triggers on past_contracts

## Code Quality

### Structure ✓
- [x] Modular architecture
- [x] Separate utility modules
- [x] Error handling
- [x] Type hints

### Testing
- [ ] Unit tests for utilities
- [ ] Integration tests for DB
- [ ] End-to-end feature tests
- [ ] Error scenario testing

## Deployment

### Before Launch
1. [ ] Install all dependencies: `pip install -r requirements.txt`
2. [ ] Verify all secrets configured in Streamlit
3. [ ] Test authentication (signup/login)
4. [ ] Test market data loading
5. [ ] Test contract management
6. [ ] Test cost calculations
7. [ ] Check for any console errors

### Deployment Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify secrets
cat .streamlit/secrets.toml

# 3. Run locally to test
streamlit run app.py

# 4. Deploy to Streamlit Cloud (optional)
# Follow https://docs.streamlit.io/deploy/streamlit-cloud
```

## Feature Verification

### Authentication
- [ ] Signup works
- [ ] Login works
- [ ] Logout works
- [ ] Session persists
- [ ] Data is user-specific

### Market Data
- [ ] Prices load from ENTSOE
- [ ] Chart displays correctly
- [ ] Moving average calculated
- [ ] CAL prices fetch/fallback work

### Contracts
- [ ] Create contracts
- [ ] Add fixations
- [ ] Update contracts
- [ ] Delete fixations
- [ ] Data persists after refresh

### Cost Analysis
- [ ] Cost calculations correct
- [ ] All breakdowns display
- [ ] Different DSOs work
- [ ] Different segments work
- [ ] VAT calculated correctly

## Performance Checklist

- [ ] First load <2 seconds
- [ ] Subsequent loads <1 second
- [ ] No console errors
- [ ] No memory leaks
- [ ] Smooth interactions

## Security Checklist

- [ ] No hardcoded secrets
- [ ] RLS prevents data leakage
- [ ] Input validation works
- [ ] Error messages don't leak info
- [ ] HTTPS enforced (if applicable)

## Documentation

- [ ] README.md complete
- [ ] IMPROVEMENTS.md current
- [ ] Code comments clear
- [ ] Setup instructions clear

## Final Steps

1. **Test Signup Flow**
   ```
   Email: test@example.com
   Password: TestPass123
   ```

2. **Test Market Data**
   - Navigate to "Marché"
   - Verify chart loads
   - Check CAL prices

3. **Test Contracts**
   - Go to "Simulation & Couverture"
   - Create a fixation
   - Verify it saved
   - Refresh page
   - Verify data persists

4. **Test Cost Analysis**
   - Go to "Coût total (réel)"
   - Select different years/DSOs
   - Verify calculations

5. **Monitor for Errors**
   - Check browser console
   - Check Streamlit terminal
   - Check Supabase logs

## Known Limitations

- Market data caches for 24 hours
- ENTSOE API rate limited (check token limits)
- Network costs based on 2026-2028 data
- No offline mode

## Support & Troubleshooting

### Common Issues

**"Missing ENTSOE_TOKEN"**
- Add token to `.streamlit/secrets.toml`
- Restart Streamlit

**"Cannot connect to database"**
- Check SUPABASE_URL and SUPABASE_ANON_KEY
- Verify Supabase project is running
- Check network connectivity

**"Authentication fails"**
- Ensure user exists in Supabase Auth
- Check password is correct
- Clear browser cookies and try again

**"No market data"**
- Wait for ENTSOE API to respond
- Check ENTSOE_TOKEN is valid
- Check date range is valid

## Rollback Plan

If issues occur:

1. Revert to backup of database (Supabase has automatic backups)
2. Check .env files for secrets
3. Review error logs
4. Contact Supabase support if needed

## Sign-Off

- [ ] All checks completed
- [ ] Team reviewed code
- [ ] Ready for production

**Launch Date**: _______________
**Deployed By**: _______________
**Notes**: _______________
