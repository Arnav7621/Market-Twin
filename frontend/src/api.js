const API = "http://127.0.0.1:8000";

async function request(
    endpoint,
    options = {}
) {
    const response = await fetch(
        `${API}${endpoint}`,
        {
            ...options,
            headers: {
                "Content-Type": "application/json",
                ...options.headers
            }
        }
    );

    const result = await response.json();

    if (!response.ok) {
        throw new Error(
            result.detail || "API request failed"
        );
    }

    return result;
}


// =========================================
// ALPHACOMPILER
// =========================================

export async function discoverAlpha(requestData) {
    return request(
        "/api/alpha/discover",
        {
            method: "POST",
            body: JSON.stringify(requestData)
        }
    );
}


// =========================================
// QUEUEMIND
// =========================================

export async function predictFill(requestData) {
    return request(
        "/api/queue-mind/predict",
        {
            method: "POST",
            body: JSON.stringify(requestData)
        }
    );
}


// =========================================
// BACKTESTING
// =========================================

export async function runBacktest(
    ticker,
    strategy
) {
    return request(
        `/api/backtest/${ticker}/${strategy}`
    );
}