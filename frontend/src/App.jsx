import { useState } from 'react'
import './App.css'

function App() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [modelType, setModelType] = useState('encoder')
  const [reportType, setReportType] = useState('klinisk')
  const [totalContainers, setTotalContainers] = useState(null)

  const handleLoadModel = async () => {
    const response = await fetch('http://localhost:5000/load_model', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model_type: modelType }),
    });
    const data = await response.json();
    setOutputText(JSON.stringify(data, null, 2));
  };

  const handleGenerate = async () => {
    const response = await fetch('http://localhost:5000/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input_text: inputText, report_type: reportType, total_containers: totalContainers }),
    })
    const data = await response.json()
    setOutputText(JSON.stringify(data, null, 2))
  }

  const handleCorrect = async () => {
    const correctedData = JSON.parse(outputText);
    const response = await fetch('http://localhost:5000/correct', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(correctedData),
    });
    const data = await response.json();
    setOutputText(JSON.stringify(data, null, 2));
  };

  return (
    <>
      <h1 style={{ textAlign: 'center', marginBottom: '1rem' }}>Labeler App</h1>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
        <div style={{ flex: 1, padding: '1rem', minWidth: '200px' }}>
          <select value={modelType} onChange={(e) => setModelType(e.target.value)} style={{ width: '100%', marginBottom: '1rem' }}>
            <option value="encoder">Encoder</option>
            <option value="encoder-decoder">Encoder-Decoder</option>
            <option value="decoder">Decoder</option>
          </select>
          <button onClick={handleLoadModel} style={{ width: '100%', marginBottom: '3rem' }}>Load Model</button>
          <select value={reportType} onChange={(e) => setReportType(e.target.value)} style={{ width: '100%', marginBottom: '1rem' }}>
            <option value="auto">Report Type (Auto)</option>
            <option value="klinisk">Klinisk</option>
            <option value="makroskopisk">Makroskopisk</option>
            <option value="mikroskopisk">Mikroskopisk</option>
          </select>
          <select value={totalContainers} onChange={(e) => setTotalContainers(e.target.value === '' ? null : Number(e.target.value))} style={{ width: '100%', marginBottom: '1rem' }}>
            <option value="">Total Containers (Auto)</option>
            {[...Array(10).keys()].map(i => (
              <option key={i + 1} value={i + 1}>{i + 1}</option>
            ))}
          </select>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your text here"
            style={{ width: '100%', height: '20vh', marginBottom: '1rem', resize: 'none' }}
          />
          <button onClick={handleGenerate} style={{ width: '100%', marginBottom: '1rem' }}>Generate</button>
        </div>
        <div style={{ flex: 1, padding: '1rem', minWidth: '200px' }}>
          <textarea
            value={outputText}
            onChange={(e) => setOutputText(e.target.value)}
            placeholder="Output will appear here"
            style={{ width: '100%', height: '40vh', marginBottom: '1rem', resize: 'none' }}
          />
          <button onClick={handleCorrect} style={{ width: '100%' }}>Correct</button>
        </div>
      </div>
    </>
  )
}

export default App
