#!/bin/bash

# Honjo Masamune Truth Engine Setup Script
# The Ultimate Truth Engine Development Environment Setup

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis for visual appeal
SWORD="⚔️"
BRAIN="🧠"
GEAR="⚙️"
CHECK="✅"
WARNING="⚠️"
ERROR="❌"
INFO="ℹ️"

echo -e "${PURPLE}${SWORD} Honjo Masamune Truth Engine Setup${NC}"
echo -e "${PURPLE}${SWORD} The Ultimate Truth Engine${NC}"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

print_error() {
    echo -e "${RED}${ERROR} $1${NC}"
}

print_info() {
    echo -e "${BLUE}${INFO} $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check system requirements
check_system_requirements() {
    print_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_status "Linux detected"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "macOS detected"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Check available memory (minimum 8GB recommended)
    if command -v free >/dev/null 2>&1; then
        MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [[ $MEMORY_GB -lt 8 ]]; then
            print_warning "Less than 8GB RAM detected. Honjo Masamune requires significant memory."
        else
            print_status "Memory check passed: ${MEMORY_GB}GB available"
        fi
    fi
    
    # Check disk space (minimum 100GB recommended)
    DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $DISK_SPACE -lt 100 ]]; then
        print_warning "Less than 100GB disk space available. Consider freeing up space."
    else
        print_status "Disk space check passed: ${DISK_SPACE}GB available"
    fi
}

# Install Rust if not present
install_rust() {
    if command -v rustc >/dev/null 2>&1; then
        print_status "Rust is already installed: $(rustc --version)"
    else
        print_info "Installing Rust..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
        print_status "Rust installed successfully"
    fi
    
    # Install required components
    print_info "Installing Rust components..."
    rustup component add clippy rustfmt
    print_status "Rust components installed"
}

# Install system dependencies
install_system_dependencies() {
    print_info "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                pkg-config \
                libssl-dev \
                libpq-dev \
                cmake \
                curl \
                git \
                jq \
                docker.io \
                docker-compose
        # RHEL/CentOS/Fedora
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y \
                gcc \
                gcc-c++ \
                make \
                pkgconfig \
                openssl-devel \
                postgresql-devel \
                cmake \
                curl \
                git \
                jq \
                docker \
                docker-compose
        else
            print_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew >/dev/null 2>&1; then
            brew install \
                cmake \
                postgresql \
                jq \
                docker \
                docker-compose
        else
            print_error "Homebrew not found. Please install Homebrew first."
            exit 1
        fi
    fi
    
    print_status "System dependencies installed"
}

# Install Cargo tools
install_cargo_tools() {
    print_info "Installing Cargo tools..."
    
    cargo install \
        cargo-audit \
        cargo-watch \
        sqlx-cli \
        mdbook
    
    print_status "Cargo tools installed"
}

# Setup Docker
setup_docker() {
    print_info "Setting up Docker..."
    
    # Start Docker service (Linux)
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo systemctl start docker
        sudo systemctl enable docker
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        print_warning "You may need to log out and back in for Docker group changes to take effect"
    fi
    
    # Test Docker
    if docker --version >/dev/null 2>&1; then
        print_status "Docker is working: $(docker --version)"
    else
        print_error "Docker installation failed"
        exit 1
    fi
    
    # Test Docker Compose
    if docker-compose --version >/dev/null 2>&1; then
        print_status "Docker Compose is working: $(docker-compose --version)"
    else
        print_error "Docker Compose installation failed"
        exit 1
    fi
}

# Create directory structure
create_directories() {
    print_info "Creating directory structure..."
    
    mkdir -p \
        data/corpus \
        data/models \
        data/dreams \
        logs \
        config/local \
        scripts/buhera \
        certs \
        backups \
        test_data
    
    print_status "Directory structure created"
}

# Setup configuration files
setup_configuration() {
    print_info "Setting up configuration files..."
    
    # Create local development config
    if [[ ! -f "config/local.yml" ]]; then
        cp config/honjo-masamune.yml config/local.yml
        
        # Modify for local development
        sed -i.bak 's/ceremonial_mode: false/ceremonial_mode: false/' config/local.yml
        sed -i.bak 's/debug_mode: true/debug_mode: true/' config/local.yml
        
        print_status "Local configuration created"
    else
        print_info "Local configuration already exists"
    fi
    
    # Create environment file
    if [[ ! -f ".env" ]]; then
        cat > .env << EOF
# Honjo Masamune Development Environment
RUST_LOG=debug
DATABASE_URL=postgresql://honjo:ceremonial_sword@localhost:5432/honjo_masamune
NEO4J_URL=bolt://localhost:7687
CLICKHOUSE_URL=http://localhost:8123
REDIS_URL=redis://localhost:6379
ATP_INITIAL_POOL=1000000
CEREMONIAL_MODE=false
EOF
        print_status "Environment file created"
    else
        print_info "Environment file already exists"
    fi
}

# Initialize databases
initialize_databases() {
    print_info "Initializing databases..."
    
    # Start database services
    docker-compose up -d postgres neo4j clickhouse redis
    
    # Wait for databases to be ready
    print_info "Waiting for databases to start..."
    sleep 30
    
    # Run database migrations
    if command -v sqlx >/dev/null 2>&1; then
        print_info "Running database migrations..."
        sqlx migrate run --database-url "postgresql://honjo:ceremonial_sword@localhost:5432/honjo_masamune"
        print_status "Database migrations completed"
    else
        print_warning "sqlx-cli not found. Database migrations skipped."
    fi
}

# Build the project
build_project() {
    print_info "Building Honjo Masamune..."
    
    # Fetch dependencies
    cargo fetch
    
    # Build in development mode
    cargo build
    
    print_status "Project built successfully"
}

# Run tests
run_tests() {
    print_info "Running tests..."
    
    # Run unit tests
    cargo test --lib
    
    print_status "Tests completed"
}

# Setup Git hooks
setup_git_hooks() {
    print_info "Setting up Git hooks..."
    
    # Create pre-commit hook
    cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Honjo Masamune pre-commit hook

echo "🔍 Running pre-commit checks..."

# Format code
cargo fmt --check
if [ $? -ne 0 ]; then
    echo "❌ Code formatting check failed. Run 'cargo fmt' to fix."
    exit 1
fi

# Run clippy
cargo clippy -- -D warnings
if [ $? -ne 0 ]; then
    echo "❌ Clippy check failed. Fix the warnings above."
    exit 1
fi

# Run tests
cargo test --lib
if [ $? -ne 0 ]; then
    echo "❌ Tests failed. Fix the failing tests."
    exit 1
fi

echo "✅ Pre-commit checks passed"
EOF
    
    chmod +x .git/hooks/pre-commit
    print_status "Git hooks configured"
}

# Generate development certificates
generate_dev_certificates() {
    print_info "Generating development certificates..."
    
    if [[ ! -f "certs/dev.pem" ]]; then
        mkdir -p certs
        
        # Generate self-signed certificate for development
        openssl req -x509 -newkey rsa:4096 -keyout certs/dev.key -out certs/dev.pem -days 365 -nodes \
            -subj "/C=US/ST=Development/L=Local/O=HonjoMasamune/OU=Development/CN=localhost"
        
        print_status "Development certificates generated"
    else
        print_info "Development certificates already exist"
    fi
}

# Main setup function
main() {
    echo -e "${CYAN}${BRAIN} Starting Honjo Masamune setup...${NC}"
    echo ""
    
    check_system_requirements
    install_rust
    install_system_dependencies
    install_cargo_tools
    setup_docker
    create_directories
    setup_configuration
    initialize_databases
    build_project
    run_tests
    setup_git_hooks
    generate_dev_certificates
    
    echo ""
    echo -e "${GREEN}${CHECK} Honjo Masamune setup completed successfully!${NC}"
    echo ""
    echo -e "${PURPLE}${SWORD} Next steps:${NC}"
    echo -e "  1. ${CYAN}make docker-compose-up${NC} - Start all services"
    echo -e "  2. ${CYAN}make health-check${NC} - Verify system health"
    echo -e "  3. ${CYAN}make truth-synthesis-demo${NC} - Run a demonstration"
    echo ""
    echo -e "${YELLOW}${WARNING} Remember: This is a development setup only${NC}"
    echo -e "${YELLOW}${WARNING} Never use ceremonial mode in development${NC}"
    echo ""
    echo -e "${PURPLE}${SWORD} The truth engine awaits...${NC}"
}

# Run setup if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi 