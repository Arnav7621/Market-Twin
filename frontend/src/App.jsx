
import { useEffect, useState } from "react";

import Plot from "react-plotly.js";

import "./App.css";

import {
  discoverAlpha,
  predictFill,
  runBacktest
} from "./api";

const API = "http://127.0.0.1:8000";


function App() {

  // =========================================
  // STATE MANAGEMENT
  // =========================================

  const [ticker, setTicker] = useState("AAPL");

  const [volatility, setVolatility] = useState("high");

  const [liquidity, setLiquidity] = useState("low");

  const [imbalance, setImbalance] = useState("negative");

  const [scenario, setScenario] = useState([]);

  const [marketData, setMarketData] = useState([]);

  const [experiments, setExperiments] = useState([]);

  const [metrics, setMetrics] = useState([]);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");

  // AlphaCompiler result
  const [alphaResult, setAlphaResult] = useState(null);

  // QueueMind result
  const [queueResult, setQueueResult] = useState(null);

  // Backtesting result
  const [backtestResult, setBacktestResult] = useState(null);

  // Research loading and errors
  const [researchLoading, setResearchLoading] = useState(false);

  const [researchError, setResearchError] = useState("");

  // Separate loading states
  const [alphaLoading, setAlphaLoading] = useState(false);

  const [queueLoading, setQueueLoading] = useState(false);

  const [backtestLoading, setBacktestLoading] = useState(false);


  // =========================================
  // FETCH MARKET DATA, EXPERIMENTS AND METRICS
  // =========================================

  useEffect(() => {

    async function fetchDashboardData() {

      setError("");

      try {

        const marketResponse = await fetch(
          `${API}/api/market-data/${ticker}`
        );

        if (!marketResponse.ok) {

          throw new Error(
            "Failed to fetch market data"
          );

        }

        const marketResult = await marketResponse.json();

        setMarketData(
          marketResult.data || []
        );


        const experimentResponse = await fetch(
          `${API}/api/experiments`
        );

        if (experimentResponse.ok) {

          const experimentResult =
            await experimentResponse.json();

          setExperiments(
            experimentResult.data || []
          );

        }


        const metricsResponse = await fetch(
          `${API}/api/metrics`
        );

        if (metricsResponse.ok) {

          const metricsResult =
            await metricsResponse.json();

          setMetrics(
            metricsResult.data || []
          );

        }

      } catch (err) {

        setError(
          err.message || "Backend is unavailable"
        );

      }

    }

    fetchDashboardData();

  }, [ticker]);


  // =========================================
  // MARKET SCENARIO GENERATOR
  // =========================================

  async function generateScenario() {

    setLoading(true);

    setError("");

    try {

      const response = await fetch(
        `${API}/api/scenario`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({

            ticker,

            volatility,

            liquidity,

            imbalance,

            duration: 1000,

            seed: 42

          })

        }
      );

      const result = await response.json();

      if (!response.ok) {

        throw new Error(
          result.detail ||
          "Scenario generation failed"
        );

      }

      setScenario(
        result.data || []
      );

    } catch (err) {

      setError(
        err.message
      );

    } finally {

      setLoading(false);

    }

  }


  // =========================================
  // ALPHACOMPILER
  // =========================================

  async function runAlphaCompiler() {

    setAlphaLoading(true);

    setResearchError("");

    setAlphaResult(null);

    try {

      const result = await discoverAlpha({

        ticker,

        population_size: 50,

        generations: 10,

        seed: 42

      });

      setAlphaResult(result);

    } catch (err) {

      setResearchError(
        `AlphaCompiler: ${err.message}`
      );

    } finally {

      setAlphaLoading(false);

    }

  }


  // =========================================
  // QUEUEMIND
  // =========================================

  async function runQueueMind() {

    setQueueLoading(true);

    setResearchError("");

    setQueueResult(null);

    try {

      const result = await predictFill({

        ticker,

        order_side: "buy",

        order_quantity: 10,

        seed: 42

      });

      setQueueResult(result);

    } catch (err) {

      setResearchError(
        `QueueMind: ${err.message}`
      );

    } finally {

      setQueueLoading(false);

    }

  }


  // =========================================
  // BACKTESTING
  // =========================================

  async function runDashboardBacktest() {

    setBacktestLoading(true);

    setResearchError("");

    setBacktestResult(null);

    try {

      const result = await runBacktest(
        ticker,
        "momentum"
      );

      setBacktestResult(result);

    } catch (err) {

      setResearchError(
        `Backtesting: ${err.message}`
      );

    } finally {

      setBacktestLoading(false);

    }

  }


  // =========================================
  // DASHBOARD UI
  // =========================================

  return (

    <div className="dashboard">

      {/* SIDEBAR */}

      <aside className="sidebar">

        <h2>MarketTwin</h2>

        <p>Quant Research Platform</p>

        <nav>

          <a href="#market">
            Market Data
          </a>

          <a href="#scenario">
            Scenario Generator
          </a>

          <a href="#alpha">
            Alpha Factors
          </a>

          <a href="#execution">
            Execution Analytics
          </a>

          <a href="#backtest">
            Backtesting
          </a>

          <a href="#validation">
            Validation
          </a>

          <a href="#experiments">
            Experiments
          </a>

        </nav>

      </aside>


      {/* MAIN DASHBOARD */}

      <main className="main">

        {/* HEADER */}

        <header className="header">

          <div>

            <h1>
              Research Dashboard
            </h1>

            <p>
              Synthetic markets, alpha discovery
              and execution research
            </p>

          </div>

          <select
            value={ticker}
            onChange={e => setTicker(e.target.value)}
          >

            <option value="AAPL">
              AAPL
            </option>

            <option value="MSFT">
              MSFT
            </option>

            <option value="NVDA">
              NVDA
            </option>

            <option value="SPY">
              SPY
            </option>

          </select>

        </header>


        {/* GLOBAL ERROR */}

        {error && (

          <p className="error">

            {error}

          </p>

        )}


        {/* RESEARCH ERROR */}

        {researchError && (

          <p className="error">

            {researchError}

          </p>

        )}


        {/* DASHBOARD CARDS */}

        <section className="cards">

          <div className="card">

            <h3>
              Market Data
            </h3>

            <p>
              {marketData.length} observations
            </p>

          </div>


          <div className="card">

            <h3>
              Experiments
            </h3>

            <p>
              {experiments.length} comparisons
            </p>

          </div>


          <div className="card">

            <h3>
              Validation
            </h3>

            <p>
              {metrics.length} metric records
            </p>

          </div>


          <div className="card">

            <h3>
              Synthetic Data
            </h3>

            <p>
              {scenario.length} observations
            </p>

          </div>

        </section>


        {/* ================================= */}
        {/* MARKET DATA EXPLORER */}
        {/* ================================= */}

        <section
          id="market"
          className="panel"
        >

          <h2>
            Market Data Explorer
          </h2>

          <Plot

            data={[
              {

                x: marketData.map(
                  d => d.Date
                ),

                y: marketData.map(
                  d => d.Close
                ),

                type: "scatter",

                mode: "lines",

                name: "Close Price"

              }
            ]}

            layout={{

              title: `${ticker} Historical Prices`,

              autosize: true,

              paper_bgcolor: "#111827",

              plot_bgcolor: "#111827",

              font: {
                color: "#ffffff"
              }

            }}

            useResizeHandler

            style={{

              width: "100%",

              height: "400px"

            }}

          />

        </section>


        {/* ================================= */}
        {/* MARKET SCENARIO GENERATOR */}
        {/* ================================= */}

        <section
          id="scenario"
          className="panel"
        >

          <h2>
            Market Scenario Generator
          </h2>

          <div className="controls">

            <label>

              Volatility

              <select
                value={volatility}
                onChange={e =>
                  setVolatility(e.target.value)
                }
              >

                <option value="high">
                  High
                </option>

                <option value="low">
                  Low
                </option>

              </select>

            </label>


            <label>

              Liquidity

              <select
                value={liquidity}
                onChange={e =>
                  setLiquidity(e.target.value)
                }
              >

                <option value="high">
                  High
                </option>

                <option value="low">
                  Low
                </option>

              </select>

            </label>


            <label>

              Imbalance

              <select
                value={imbalance}
                onChange={e =>
                  setImbalance(e.target.value)
                }
              >

                <option value="positive">
                  Positive
                </option>

                <option value="negative">
                  Negative
                </option>

                <option value="neutral">
                  Neutral
                </option>

              </select>

            </label>


            <button
              onClick={generateScenario}
              disabled={loading}
            >

              {loading
                ? "Generating..."
                : "Generate Scenario"}

            </button>

          </div>


          {scenario.length > 0 && (

            <Plot

              data={[
                {

                  x: scenario.map(
                    d => d.Step
                  ),

                  y: scenario.map(
                    d => d.Price
                  ),

                  type: "scatter",

                  mode: "lines",

                  name: "Synthetic Price"

                }
              ]}

              layout={{

                title: "Synthetic Market Price",

                autosize: true,

                paper_bgcolor: "#111827",

                plot_bgcolor: "#111827",

                font: {
                  color: "#ffffff"
                }

              }}

              useResizeHandler

              style={{

                width: "100%",

                height: "400px"

              }}

            />

          )}

        </section>


        {/* ================================= */}
        {/* ALPHACOMPILER */}
        {/* ================================= */}

        <section
          id="alpha"
          className="panel"
        >

          <h2>
            AlphaCompiler — Factor Discovery
          </h2>

          <p>

            Discover candidate alpha factors using
            the symbolic research pipeline.

          </p>

          <button
            onClick={runAlphaCompiler}
            disabled={alphaLoading}
          >

            {alphaLoading
              ? "Discovering Factors..."
              : "Run AlphaCompiler"}

          </button>


          {alphaResult && (

            <div className="result">

              <h3>
                AlphaCompiler Results
              </h3>

              <pre>

                {JSON.stringify(
                  alphaResult,
                  null,
                  2
                )}

              </pre>

            </div>

          )}

        </section>


        {/* ================================= */}
        {/* QUEUEMIND EXECUTION ANALYTICS */}
        {/* ================================= */}

        <section
          id="execution"
          className="panel"
        >

          <h2>
            QueueMind — Execution Analytics
          </h2>

          <p>

            Predict order fill probability using
            the QueueMind prediction pipeline.

          </p>

          <div className="controls">

            <label>

              Ticker

              <input
                type="text"
                value={ticker}
                readOnly
              />

            </label>

            <label>

              Order Side

              <input
                type="text"
                value="BUY"
                readOnly
              />

            </label>

            <label>

              Order Quantity

              <input
                type="number"
                value={10}
                readOnly
              />

            </label>

          </div>


          <button
            onClick={runQueueMind}
            disabled={queueLoading}
          >

            {queueLoading
              ? "Predicting Fill Probability..."
              : "Run QueueMind"}

          </button>


          {queueResult && (

            <div className="result">

              <h3>
                QueueMind Results
              </h3>

              <pre>

                {JSON.stringify(
                  queueResult,
                  null,
                  2
                )}

              </pre>

            </div>

          )}

        </section>


        {/* ================================= */}
        {/* STRATEGY BACKTESTING */}
        {/* ================================= */}

        <section
          id="backtest"
          className="panel"
        >

          <h2>
            Strategy Backtesting
          </h2>

          <p>

            Evaluate the momentum strategy using
            historical market data.

          </p>


          <div className="controls">

            <label>

              Ticker

              <input
                type="text"
                value={ticker}
                readOnly
              />

            </label>

            <label>

              Strategy

              <input
                type="text"
                value="Momentum"
                readOnly
              />

            </label>

          </div>


          <button
            onClick={runDashboardBacktest}
            disabled={backtestLoading}
          >

            {backtestLoading
              ? "Running Backtest..."
              : "Run Backtest"}

          </button>


          {backtestResult && (

            <div className="result">

              <h3>
                Backtest Results
              </h3>

              <pre>

                {JSON.stringify(
                  backtestResult.summary ??
                  backtestResult,
                  null,
                  2
                )}

              </pre>

            </div>

          )}

        </section>


        {/* ================================= */}
        {/* STATISTICAL VALIDATION */}
        {/* ================================= */}

        <section
          id="validation"
          className="panel"
        >

          <h2>
            Statistical Validation
          </h2>

          {metrics.length === 0 && (

            <p>
              No validation metrics available.
            </p>

          )}

          {metrics.map(
            (metric, index) => (

              <pre key={index}>

                {JSON.stringify(
                  metric,
                  null,
                  2
                )}

              </pre>

            )
          )}

        </section>


        {/* ================================= */}
        {/* EXPERIMENT COMPARISON */}
        {/* ================================= */}

        <section
          id="experiments"
          className="panel"
        >

          <h2>
            Experiment Comparison
          </h2>

          {experiments.length === 0 && (

            <p>
              No experiment results available.
            </p>

          )}


          {experiments.length > 0 && (

            <Plot

              data={[
                {

                  x: experiments.map(
                    d => d.Experiment
                  ),

                  y: experiments.map(
                    d => d.IC
                  ),

                  type: "bar",

                  name: "Information Coefficient"

                }
              ]}

              layout={{

                title: "Ablation Experiment IC",

                autosize: true,

                paper_bgcolor: "#111827",

                plot_bgcolor: "#111827",

                font: {
                  color: "#ffffff"
                }

              }}

              useResizeHandler

              style={{

                width: "100%",

                height: "400px"

              }}

            />

          )}

        </section>

      </main>

    </div>

  );

}


export default App;
