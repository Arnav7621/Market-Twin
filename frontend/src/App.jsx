
import { useEffect, useState } from "react";

import Plot from "react-plotly.js";

import "./App.css";

const API = "http://127.0.0.1:8000";


function App() {

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


  useEffect(() => {

    fetch(`${API}/api/market-data/${ticker}`)
      .then(res => res.json())
      .then(data => setMarketData(data.data || []))
      .catch(() => setError("Backend is unavailable"));

    fetch(`${API}/api/experiments`)
      .then(res => res.json())
      .then(data => setExperiments(data.data || []))
      .catch(() => { });

    fetch(`${API}/api/metrics`)
      .then(res => res.json())
      .then(data => setMetrics(data.data || []))
      .catch(() => { });

  }, [ticker]);


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
          result.detail || "Scenario generation failed"
        );

      }

      setScenario(result.data || []);

    } catch (err) {

      setError(err.message);

    } finally {

      setLoading(false);

    }

  }


  return (

    <div className="dashboard">

      <aside className="sidebar">

        <h2>MarketTwin</h2>

        <p>Quant Research Platform</p>

        <nav>

          <a href="#market">Market Data</a>

          <a href="#scenario">Scenario Generator</a>

          <a href="#alpha">Alpha Factors</a>

          <a href="#execution">Execution Analytics</a>

          <a href="#backtest">Backtesting</a>

          <a href="#validation">Validation</a>

          <a href="#experiments">Experiments</a>

        </nav>

      </aside>


      <main className="main">

        <header className="header">

          <div>

            <h1>Research Dashboard</h1>

            <p>
              Synthetic markets, alpha discovery
              and execution research
            </p>

          </div>

          <select
            value={ticker}
            onChange={e => setTicker(e.target.value)}
          >

            <option value="AAPL">AAPL</option>

            <option value="MSFT">MSFT</option>

            <option value="NVDA">NVDA</option>

            <option value="SPY">SPY</option>

          </select>

        </header>


        {error && (

          <p className="error">
            {error}
          </p>

        )}


        <section className="cards">

          <div className="card">

            <h3>Market Data</h3>

            <p>{marketData.length} observations</p>

          </div>

          <div className="card">

            <h3>Experiments</h3>

            <p>{experiments.length} comparisons</p>

          </div>

          <div className="card">

            <h3>Validation</h3>

            <p>{metrics.length} metric records</p>

          </div>

          <div className="card">

            <h3>Synthetic Data</h3>

            <p>{scenario.length} observations</p>

          </div>

        </section>


        <section id="market" className="panel">

          <h2>Market Data Explorer</h2>

          <Plot

            data={[
              {
                x: marketData.map(d => d.Date),

                y: marketData.map(d => d.Close),

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


        <section id="scenario" className="panel">

          <h2>Market Scenario Generator</h2>

          <div className="controls">

            <label>

              Volatility

              <select
                value={volatility}
                onChange={e => setVolatility(e.target.value)}
              >

                <option value="high">High</option>

                <option value="low">Low</option>

              </select>

            </label>


            <label>

              Liquidity

              <select
                value={liquidity}
                onChange={e => setLiquidity(e.target.value)}
              >

                <option value="high">High</option>

                <option value="low">Low</option>

              </select>

            </label>


            <label>

              Imbalance

              <select
                value={imbalance}
                onChange={e => setImbalance(e.target.value)}
              >

                <option value="positive">Positive</option>

                <option value="negative">Negative</option>

                <option value="neutral">Neutral</option>

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
                  x: scenario.map(d => d.Step),

                  y: scenario.map(d => d.Price),

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


        <section id="alpha" className="panel">

          <h2>Alpha Factor Discovery</h2>

          <p>
            Factor discovery output will be displayed
            after the AlphaCompiler pipeline is connected.
          </p>

        </section>


        <section id="execution" className="panel">

          <h2>Execution Analytics</h2>

          <p>
            Fill probability, spread and slippage
            analytics.
          </p>

        </section>


        <section id="backtest" className="panel">

          <h2>Strategy Backtesting</h2>

          <p>
            Strategy PnL and backtest results.
          </p>

        </section>


        <section id="validation" className="panel">

          <h2>Statistical Validation</h2>

          {metrics.map((metric, index) => (

            <pre key={index}>

              {JSON.stringify(metric, null, 2)}

            </pre>

          ))}

        </section>


        <section id="experiments" className="panel">

          <h2>Experiment Comparison</h2>

          {experiments.length > 0 && (

            <Plot

              data={[
                {
                  x: experiments.map(d => d.Experiment),

                  y: experiments.map(d => d.IC),

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
