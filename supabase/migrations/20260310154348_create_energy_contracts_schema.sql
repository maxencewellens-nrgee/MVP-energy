/*
  # Energy Contracts Management Schema

  1. New Tables
    - `contracts`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `year` (integer, contract year)
      - `total_mwh` (numeric, total energy volume)
      - `max_fixations` (integer, maximum allowed fixations)
      - `dso` (text, energy distributor)
      - `segment` (text, BT or MT)
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)
    
    - `fixations`
      - `id` (uuid, primary key)
      - `contract_id` (uuid, references contracts)
      - `user_id` (uuid, references auth.users)
      - `date` (date, fixation date)
      - `price` (numeric, price per MWh in EUR)
      - `volume` (numeric, volume in MWh)
      - `created_at` (timestamptz)
    
    - `past_contracts`
      - `id` (uuid, primary key)
      - `user_id` (uuid, references auth.users)
      - `year` (integer)
      - `fixed_volume` (numeric)
      - `fixed_price` (numeric)
      - `created_at` (timestamptz)
      - `updated_at` (timestamptz)
    
    - `market_data_cache`
      - `id` (uuid, primary key)
      - `date` (date, unique)
      - `avg_price` (numeric)
      - `min_price` (numeric)
      - `max_price` (numeric)
      - `data_points` (integer)
      - `created_at` (timestamptz)

  2. Security
    - Enable RLS on all tables
    - Add policies for authenticated users to manage their own data
*/

-- Create contracts table
CREATE TABLE IF NOT EXISTS contracts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  year integer NOT NULL CHECK (year >= 2024 AND year <= 2050),
  total_mwh numeric NOT NULL CHECK (total_mwh >= 0) DEFAULT 200.0,
  max_fixations integer NOT NULL CHECK (max_fixations >= 1 AND max_fixations <= 50) DEFAULT 5,
  dso text NOT NULL DEFAULT 'ORES',
  segment text NOT NULL DEFAULT 'BT',
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  UNIQUE(user_id, year)
);

ALTER TABLE contracts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own contracts"
  ON contracts FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own contracts"
  ON contracts FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own contracts"
  ON contracts FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own contracts"
  ON contracts FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Create fixations table
CREATE TABLE IF NOT EXISTS fixations (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  contract_id uuid REFERENCES contracts(id) ON DELETE CASCADE NOT NULL,
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  date date NOT NULL DEFAULT CURRENT_DATE,
  price numeric NOT NULL CHECK (price >= 0),
  volume numeric NOT NULL CHECK (volume >= 0),
  created_at timestamptz DEFAULT now()
);

ALTER TABLE fixations ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own fixations"
  ON fixations FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own fixations"
  ON fixations FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own fixations"
  ON fixations FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own fixations"
  ON fixations FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Create past_contracts table
CREATE TABLE IF NOT EXISTS past_contracts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  year integer NOT NULL CHECK (year >= 2020 AND year <= 2030),
  fixed_volume numeric NOT NULL CHECK (fixed_volume >= 0) DEFAULT 0,
  fixed_price numeric NOT NULL CHECK (fixed_price >= 0) DEFAULT 0,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  UNIQUE(user_id, year)
);

ALTER TABLE past_contracts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own past contracts"
  ON past_contracts FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own past contracts"
  ON past_contracts FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own past contracts"
  ON past_contracts FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own past contracts"
  ON past_contracts FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

-- Create market_data_cache table (public read, no user_id)
CREATE TABLE IF NOT EXISTS market_data_cache (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  date date UNIQUE NOT NULL,
  avg_price numeric NOT NULL,
  min_price numeric NOT NULL,
  max_price numeric NOT NULL,
  data_points integer NOT NULL DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

ALTER TABLE market_data_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can view market data"
  ON market_data_cache FOR SELECT
  TO authenticated
  USING (true);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_contracts_user_year ON contracts(user_id, year);
CREATE INDEX IF NOT EXISTS idx_fixations_contract ON fixations(contract_id);
CREATE INDEX IF NOT EXISTS idx_fixations_user ON fixations(user_id);
CREATE INDEX IF NOT EXISTS idx_past_contracts_user_year ON past_contracts(user_id, year);
CREATE INDEX IF NOT EXISTS idx_market_data_date ON market_data_cache(date);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_contracts_updated_at') THEN
    CREATE TRIGGER update_contracts_updated_at
      BEFORE UPDATE ON contracts
      FOR EACH ROW
      EXECUTE FUNCTION update_updated_at_column();
  END IF;

  IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'update_past_contracts_updated_at') THEN
    CREATE TRIGGER update_past_contracts_updated_at
      BEFORE UPDATE ON past_contracts
      FOR EACH ROW
      EXECUTE FUNCTION update_updated_at_column();
  END IF;
END $$;