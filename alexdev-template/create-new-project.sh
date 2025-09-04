#!/bin/bash

# Create New Startup Project Script
# Usage: ./create-new-project.sh <project-name>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <project-name>"
    echo "Example: $0 alexdev-gemini-microservice"
    exit 1
fi

PROJECT_NAME="$1"
NEW_PROJECT_PATH="../$PROJECT_NAME"

echo "Creating new startup project: $PROJECT_NAME"
echo "Target path: $NEW_PROJECT_PATH"

# Check if target directory already exists
if [ -d "$NEW_PROJECT_PATH" ]; then
    echo "Error: Directory $NEW_PROJECT_PATH already exists"
    exit 1
fi

# Create new project directory
mkdir -p "$NEW_PROJECT_PATH"

# Copy specific files to project root
echo "Copying framework files to project root..."
cp "alexdev-template/STARTUP_ROADMAP.json" "$NEW_PROJECT_PATH/"
cp "alexdev-template/CLAUDE.md" "$NEW_PROJECT_PATH/"
cp -r "alexdev-template/references" "$NEW_PROJECT_PATH/"

# Copy entire template to preserve for future use
echo "Copying complete template for preservation..."
cp -r "alexdev-template" "$NEW_PROJECT_PATH/"

# Update project metadata in root STARTUP_ROADMAP.json
echo "Updating project metadata..."
CURRENT_DATE=$(date +%Y-%m-%d)

# Use sed to update project name and date (works on both Linux and macOS)
sed -i.bak "s/\"name\": \".*\"/\"name\": \"$PROJECT_NAME\"/" "$NEW_PROJECT_PATH/STARTUP_ROADMAP.json"
sed -i.bak "s/\"last_updated\": \".*\"/\"last_updated\": \"$CURRENT_DATE\"/" "$NEW_PROJECT_PATH/STARTUP_ROADMAP.json"

# Remove backup file created by sed
rm "$NEW_PROJECT_PATH/STARTUP_ROADMAP.json.bak"

echo "✅ New startup project created successfully!"
echo "📁 Project location: $NEW_PROJECT_PATH"
echo "🚀 Ready to start Phase 1: Concept & Validation"
echo ""
echo "Next steps:"
echo "1. cd $NEW_PROJECT_PATH"
echo "2. Review STARTUP_ROADMAP.json"
echo "3. Begin Phase 1 with Claude assistance"