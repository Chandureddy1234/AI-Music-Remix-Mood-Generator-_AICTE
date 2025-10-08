@echo off
echo ============================================
echo  AI Music Generator - GitHub Upload Script
echo ============================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

echo [1/6] Initializing Git repository...
git init
if errorlevel 1 (
    echo Git already initialized or error occurred.
)

echo.
echo [2/6] Adding remote repository...
git remote remove origin 2>nul
git remote add origin https://github.com/Chandureddy1234/AI-Music-Remix-Mood-Generator-_AICTE.git
if errorlevel 1 (
    echo ERROR: Failed to add remote repository!
    pause
    exit /b 1
)

echo.
echo [3/6] Checking current branch...
git branch -M main

echo.
echo [4/6] Staging all files...
git add .
if errorlevel 1 (
    echo ERROR: Failed to stage files!
    pause
    exit /b 1
)

echo.
echo [5/6] Creating commit...
git commit -m "Initial commit: AI Music Remix & Mood Generator - Complete Project"
if errorlevel 1 (
    echo Note: No changes to commit or commit failed.
)

echo.
echo [6/6] Pushing to GitHub...
echo.
echo NOTE: You will be prompted for your GitHub credentials.
echo If you have 2FA enabled, use a Personal Access Token instead of password.
echo.
echo To create a token:
echo 1. Go to: https://github.com/settings/tokens
echo 2. Click "Generate new token (classic)"
echo 3. Select "repo" scope
echo 4. Copy the token and use it as your password
echo.
pause

git push -u origin main
if errorlevel 1 (
    echo.
    echo ERROR: Push failed!
    echo.
    echo Common solutions:
    echo 1. Make sure you're logged into GitHub
    echo 2. If using 2FA, use a Personal Access Token instead of password
    echo 3. Check your internet connection
    echo 4. Make sure the repository URL is correct
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  SUCCESS! Project uploaded to GitHub!
echo ============================================
echo.
echo Your repository: https://github.com/Chandureddy1234/AI-Music-Remix-Mood-Generator-_AICTE
echo.
echo Next steps:
echo 1. Visit your repository on GitHub
echo 2. Verify all files are uploaded
echo 3. Deploy to Streamlit Cloud (optional)
echo.
pause
