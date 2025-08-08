# Crypto Trading Analysis System

## Project Overview

This project is a comprehensive cryptocurrency trading analysis system. It provides a web-based dashboard to visualize market data, sentiment analysis, and potential trading opportunities for leveraged perpetual contracts. The system is composed of a Python backend powered by Flask and a React frontend.

The backend is responsible for:
-   Collecting real-time and historical data from various sources (Binance, Yahoo Finance, etc.).
-   Performing in-depth technical analysis (e.g., moving averages, RSI, volume profile).
-   Conducting sentiment analysis (e.g., funding rates, open interest).
-   Identifying high-probability trading setups based on a set of pre-defined strategies.
-   Generating detailed trading reports.

The frontend is a user-friendly dashboard that displays:
-   A market overview with overall sentiment.
-   A list of actionable trading opportunities with entry/exit points and risk metrics.
-   Detailed analysis for individual assets.

## Building and Running

### Backend

The backend is a Flask application. To run it, you need to have Python and the required dependencies installed.

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt 
    ```
    *(Note: a `requirements.txt` file is not present, but would be the standard way to manage dependencies)*

2.  **Run the Flask server:**
    ```bash
    python main.py
    ```
    The backend API will be available at `http://localhost:5001`.

### Frontend

The frontend is a React application.

1.  **Install dependencies:**
    ```bash
    npm install
    ```
    *(Note: a `package.json` file is not present, but would be the standard way to manage dependencies)*

2.  **Run the development server:**
    ```bash
    npm run dev
    ```
    The frontend will be available at `http://localhost:5173` by default and will proxy API requests to the backend.

### Demo

The project includes a command-line demo script that showcases the system's capabilities without running the web interface.

```bash
python crypto_trading_system_demo.py
```

## Development Conventions

-   **Backend:** The backend follows a modular structure, with clear separation of concerns for data collection, analysis, and API endpoints. It uses Flask Blueprints to organize routes.
-   **Frontend:** The frontend is built with React and uses functional components with hooks. It interacts with the backend through a REST API.
-   **Analysis:** The analysis modules are designed to be extensible, allowing for the addition of new data sources, technical indicators, and trading strategies.
-   **Configuration:** Key parameters like API keys and trading settings should be managed through environment variables or a configuration file (not currently implemented).
