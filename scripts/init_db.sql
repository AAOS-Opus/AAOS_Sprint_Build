-- AAOS Production Database Initialization Script
-- This script runs on first PostgreSQL startup

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges (if running as superuser)
GRANT ALL PRIVILEGES ON DATABASE aaos_prod TO aaos_user;

-- Note: Schema creation is handled by Alembic migrations
-- Run: alembic upgrade head after container startup
