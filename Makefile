# Honjo Masamune Truth Engine Makefile
# The Ultimate Truth Engine Build System

.PHONY: help build test clean docker setup-dev ceremonial-deploy

# Default target
help: ## Show this help message
	@echo "🗾 Honjo Masamune Truth Engine Build System"
	@echo "⚔️  The Ultimate Truth Engine"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
build: ## Build the Honjo Masamune engine
	@echo "🔨 Building Honjo Masamune Truth Engine..."
	cargo build --release
	@echo "✅ Build complete"

build-dev: ## Build in development mode
	@echo "🔨 Building Honjo Masamune (development mode)..."
	cargo build
	@echo "✅ Development build complete"

build-all: ## Build all binaries
	@echo "🔨 Building all Honjo Masamune binaries..."
	cargo build --release --bin honjo-masamune
	cargo build --release --bin buhera-cli
	cargo build --release --bin preparation-manager
	cargo build --release --bin truth-synthesizer
	@echo "✅ All binaries built"

# Test targets
test: ## Run all tests
	@echo "🧪 Running Honjo Masamune tests..."
	cargo test
	@echo "✅ Tests complete"

test-fuzzy: ## Run fuzzy logic tests specifically
	@echo "🧪 Running fuzzy logic tests..."
	cargo test --package fuzzy-logic-core
	@echo "✅ Fuzzy logic tests complete"

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	cargo test --test integration_tests
	@echo "✅ Integration tests complete"

test-ceremonial: ## Run ceremonial mode tests (requires special setup)
	@echo "⚔️  Running ceremonial mode tests..."
	@echo "⚠️  WARNING: These tests simulate ceremonial operations"
	cargo test --features ceremonial-mode ceremonial_tests
	@echo "✅ Ceremonial tests complete"

# Benchmarks
bench: ## Run performance benchmarks
	@echo "📊 Running performance benchmarks..."
	cargo bench
	@echo "✅ Benchmarks complete"

bench-atp: ## Benchmark ATP metabolism system
	@echo "📊 Benchmarking ATP metabolism..."
	cargo bench --package atp-manager
	@echo "✅ ATP benchmarks complete"

# Code quality
lint: ## Run linting
	@echo "🔍 Running linting..."
	cargo clippy -- -D warnings
	@echo "✅ Linting complete"

fmt: ## Format code
	@echo "🎨 Formatting code..."
	cargo fmt
	@echo "✅ Code formatted"

audit: ## Security audit
	@echo "🔒 Running security audit..."
	cargo audit
	@echo "✅ Security audit complete"

# Docker targets
docker-build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	docker build -t honjo-masamune:latest .
	@echo "✅ Docker image built"

docker-build-all: ## Build all Docker images
	@echo "🐳 Building all Docker images..."
	docker build -t honjo-masamune:latest .
	docker build -f Dockerfile.preparation -t honjo-preparation:latest .
	docker build -f Dockerfile.dreaming -t honjo-dreaming:latest .
	@echo "✅ All Docker images built"

docker-run: ## Run Docker container
	@echo "🐳 Running Honjo Masamune container..."
	docker run -p 8080:8080 -p 8081:8081 -p 8082:8082 honjo-masamune:latest

docker-compose-up: ## Start all services with Docker Compose
	@echo "🐳 Starting Honjo Masamune infrastructure..."
	docker-compose up -d
	@echo "✅ Infrastructure started"
	@echo "🌐 API available at http://localhost:8080"
	@echo "📊 Grafana available at http://localhost:3000"
	@echo "🔍 Jaeger available at http://localhost:16686"

docker-compose-down: ## Stop all services
	@echo "🐳 Stopping Honjo Masamune infrastructure..."
	docker-compose down
	@echo "✅ Infrastructure stopped"

# Development setup
setup-dev: ## Set up development environment
	@echo "🛠️  Setting up development environment..."
	@echo "📦 Installing Rust dependencies..."
	cargo fetch
	@echo "🔧 Installing development tools..."
	rustup component add clippy rustfmt
	cargo install cargo-audit cargo-watch
	@echo "📁 Creating necessary directories..."
	mkdir -p data/corpus data/models data/dreams logs
	@echo "🗄️  Setting up databases..."
	docker-compose up -d postgres neo4j clickhouse redis
	@echo "⏳ Waiting for databases to be ready..."
	sleep 10
	@echo "🗃️  Running database migrations..."
	cargo run --bin setup-databases
	@echo "✅ Development environment ready"

setup-monitoring: ## Set up monitoring stack
	@echo "📊 Setting up monitoring..."
	docker-compose up -d prometheus grafana jaeger
	@echo "✅ Monitoring stack ready"

# Database management
db-migrate: ## Run database migrations
	@echo "🗃️  Running database migrations..."
	sqlx migrate run --database-url postgresql://honjo:ceremonial_sword@localhost:5432/honjo_masamune
	@echo "✅ Migrations complete"

db-reset: ## Reset databases (WARNING: Destructive)
	@echo "⚠️  WARNING: This will destroy all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "🗃️  Resetting databases..."; \
		docker-compose down -v; \
		docker-compose up -d postgres neo4j clickhouse redis; \
		sleep 10; \
		make db-migrate; \
		echo "✅ Databases reset"; \
	else \
		echo ""; \
		echo "❌ Database reset cancelled"; \
	fi

# Preparation and corpus management
prepare-corpus: ## Prepare information corpus
	@echo "📚 Preparing information corpus..."
	cargo run --bin preparation-manager -- --corpus-path ./data/corpus
	@echo "✅ Corpus preparation complete"

validate-corpus: ## Validate corpus integrity
	@echo "🔍 Validating corpus integrity..."
	cargo run --bin preparation-manager -- --validate-only
	@echo "✅ Corpus validation complete"

# Buhera script management
buhera-compile: ## Compile Buhera scripts
	@echo "📜 Compiling Buhera scripts..."
	cargo run --bin buhera-cli -- compile scripts/
	@echo "✅ Buhera scripts compiled"

buhera-test: ## Test Buhera scripts
	@echo "🧪 Testing Buhera scripts..."
	cargo run --bin buhera-cli -- test scripts/
	@echo "✅ Buhera script tests complete"

# Ceremonial deployment (Production only)
ceremonial-check: ## Check ceremonial readiness
	@echo "⚔️  Checking ceremonial readiness..."
	@echo "🔍 Verifying elite organization credentials..."
	@echo "💰 Checking financial capability requirements..."
	@echo "🧠 Assessing intellectual sophistication..."
	@echo "⚖️  Validating moral authority..."
	cargo run --bin honjo-masamune -- --config config/ceremonial.yml --check-readiness
	@echo "✅ Ceremonial readiness check complete"

ceremonial-deploy: ## Deploy in ceremonial mode (PRODUCTION ONLY)
	@echo "⚔️  WARNING: CEREMONIAL DEPLOYMENT"
	@echo "⚔️  This deployment will enable the legendary sword"
	@echo "⚔️  Each use permanently closes discussion on topics"
	@echo "⚔️  There is no going back"
	@echo ""
	@read -p "Do you have the authority to draw the ceremonial sword? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "⚔️  Deploying Honjo Masamune in ceremonial mode..."; \
		docker-compose -f docker-compose.ceremonial.yml up -d; \
		echo "⚔️  CEREMONIAL MODE ACTIVE"; \
		echo "⚔️  The sword has been drawn"; \
	else \
		echo ""; \
		echo "❌ Ceremonial deployment cancelled"; \
		echo "⚔️  The sword remains sheathed"; \
	fi

# Monitoring and maintenance
logs: ## View logs
	@echo "📋 Viewing Honjo Masamune logs..."
	docker-compose logs -f honjo-masamune

logs-all: ## View all service logs
	@echo "📋 Viewing all service logs..."
	docker-compose logs -f

health-check: ## Check system health
	@echo "🏥 Checking system health..."
	curl -f http://localhost:8080/health || echo "❌ System unhealthy"
	@echo "✅ Health check complete"

metrics: ## View system metrics
	@echo "📊 Opening metrics dashboard..."
	@echo "🌐 Grafana: http://localhost:3000"
	@echo "📈 Prometheus: http://localhost:9090"
	@echo "🔍 Jaeger: http://localhost:16686"

# Cleanup
clean: ## Clean build artifacts
	@echo "🧹 Cleaning build artifacts..."
	cargo clean
	@echo "✅ Clean complete"

clean-docker: ## Clean Docker images and volumes
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✅ Docker cleanup complete"

clean-all: clean clean-docker ## Clean everything
	@echo "🧹 Deep cleaning..."
	rm -rf target/
	rm -rf data/models/*
	rm -rf data/dreams/*
	rm -rf logs/*
	@echo "✅ Deep clean complete"

# Documentation
docs: ## Generate documentation
	@echo "📚 Generating documentation..."
	cargo doc --no-deps --open
	@echo "✅ Documentation generated"

docs-book: ## Generate mdBook documentation
	@echo "📖 Generating mdBook documentation..."
	mdbook build docs/
	@echo "✅ Book documentation generated"

# Release management
release-check: ## Check release readiness
	@echo "🔍 Checking release readiness..."
	make test
	make lint
	make audit
	@echo "✅ Release checks complete"

release-build: ## Build release artifacts
	@echo "📦 Building release artifacts..."
	cargo build --release
	@echo "🗜️  Creating release archive..."
	tar -czf honjo-masamune-$(shell cargo metadata --format-version 1 | jq -r '.packages[0].version').tar.gz \
		target/release/honjo-masamune \
		target/release/buhera-cli \
		target/release/preparation-manager \
		target/release/truth-synthesizer \
		config/ \
		README.md \
		LICENSE.md
	@echo "✅ Release artifacts created"

# Special targets
truth-synthesis-demo: ## Run truth synthesis demonstration
	@echo "🔍 Running truth synthesis demonstration..."
	@echo "⚠️  This is a demonstration only - not ceremonial use"
	cargo run --bin truth-synthesizer -- --demo-mode --query "What is the nature of truth?"
	@echo "✅ Demonstration complete"

dream-cycle-demo: ## Run dreaming cycle demonstration
	@echo "💭 Running dreaming cycle demonstration..."
	cargo run --bin honjo-masamune -- --config config/demo.yml --dream-cycle-only
	@echo "✅ Dream cycle demonstration complete"

# Emergency procedures
emergency-shutdown: ## Emergency shutdown of all services
	@echo "🚨 EMERGENCY SHUTDOWN"
	docker-compose down --remove-orphans
	pkill -f honjo-masamune || true
	@echo "🛑 Emergency shutdown complete"

ceremonial-emergency-stop: ## Emergency stop of ceremonial mode
	@echo "🚨 CEREMONIAL EMERGENCY STOP"
	@echo "⚔️  Forcibly sheathing the sword..."
	docker-compose -f docker-compose.ceremonial.yml down --remove-orphans
	@echo "⚔️  Ceremonial mode terminated"
	@echo "⚔️  The sword has been forcibly sheathed"

# Version information
version: ## Show version information
	@echo "🗾 Honjo Masamune Truth Engine"
	@echo "Version: $(shell cargo metadata --format-version 1 | jq -r '.packages[0].version')"
	@echo "Build: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
	@echo "Rust: $(shell rustc --version)"
	@echo "⚔️  The Ultimate Truth Engine" 