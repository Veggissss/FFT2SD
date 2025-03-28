import { useState } from 'react'
import Editor from '@monaco-editor/react'
import './App.css'

function App() {
  const [inputText, setInputText] = useState('')
  const [outputText, setOutputText] = useState('')
  const [modelType, setModelType] = useState('encoder')
  const [reportType, setReportType] = useState('auto')
  const [totalContainers, setTotalContainers] = useState(null)
  const [jsonList, setJsonList] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);


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
    });
    const data = await response.json();
    setJsonList(data);
    setCurrentIndex(0);
    setOutputText(JSON.stringify(data[0], null, 2));
  };

  const handleNext = () => {
    if (currentIndex < jsonList.length - 1) {
      // Update the current JSON in the list with any changes from the output text field
      try {
        const updatedJsonList = [...jsonList];
        updatedJsonList[currentIndex] = JSON.parse(outputText);
        setJsonList(updatedJsonList);

        const newIndex = currentIndex + 1;
        setCurrentIndex(newIndex);
        setOutputText(JSON.stringify(updatedJsonList[newIndex], null, 2));
      } catch (error) {
        alert('Invalid JSON format. Please correct before proceeding.');
      }
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      // Update the current JSON in the list with any changes from the output text field
      try {
        const updatedJsonList = [...jsonList];
        updatedJsonList[currentIndex] = JSON.parse(outputText);
        setJsonList(updatedJsonList);

        const newIndex = currentIndex - 1;
        setCurrentIndex(newIndex);
        setOutputText(JSON.stringify(updatedJsonList[newIndex], null, 2));
      } catch (error) {
        alert('Invalid JSON format. Please correct before proceeding.');
      }
    }
  };

  const handleCorrect = async () => {
    // Update the current JSON in the list with any changes from the output text field
    try {
      const updatedJsonList = [...jsonList];
      updatedJsonList[currentIndex] = JSON.parse(outputText);
      setJsonList(updatedJsonList);

      // Send the entire updated list to the '/correct' endpoint
      const response = await fetch('http://localhost:5000/correct', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updatedJsonList),
      });

      // Clear the jsonList and show the response
      setJsonList([]);
      const data = await response.json();
      setOutputText(JSON.stringify(data, null, 2));
    } catch (error) {
      alert('Invalid JSON format. Please correct before proceeding.');
    }
  };

  return (
    <>
      <h1 className="app-title">Pathology Report Labeler</h1>
      <div className="model-panel">
        <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
          <option value="encoder">Encoder</option>
          <option value="encoder-decoder">Encoder-Decoder</option>
          <option value="decoder">Decoder</option>
        </select>
        <button onClick={handleLoadModel} className="action-button">Load Model</button>
      </div>
      <div className="app-container">
        <div className="left-panel">
          <select value={reportType} onChange={(e) => setReportType(e.target.value)}>
            <option value="auto">Report Type (Auto)</option>
            <option value="klinisk">Klinisk</option>
            <option value="makroskopisk">Makroskopisk</option>
            <option value="mikroskopisk">Mikroskopisk</option>
          </select>
          <select value={totalContainers} onChange={(e) => setTotalContainers(e.target.value === '' ? null : Number(e.target.value))}>
            <option value="">Total Containers (Auto)</option>
            {[...Array(10).keys()].map(i => (
              <option key={i + 1} value={i + 1}>{i + 1}</option>
            ))}
          </select>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Report text..."
            className="input-textarea"
          />
          <button onClick={handleGenerate} className="action-button">Generate</button>
        </div>
        <div className="right-panel">
          <div className="editor-container">
            <Editor
              height="100%"
              defaultLanguage="json"
              value={outputText}
              onChange={setOutputText}
              theme="vs-dark"
              options={{
                minimap: { enabled: false },
                fontSize: 14,
                wordWrap: 'on',
                scrollBeyondLastLine: false,
                automaticLayout: true
              }}
            />
          </div>
          <div className="navigation-controls">
            <button onClick={handlePrevious} disabled={currentIndex === 0 || jsonList.length === 0}>Prev</button>
            <span>{jsonList.length > 0 ? `${currentIndex + 1} / ${jsonList.length}` : 'No data'}</span>
            <button onClick={handleNext} disabled={currentIndex === jsonList.length - 1 || jsonList.length === 0}>Next</button>
          </div>
          <button onClick={handleCorrect} className="action-button" disabled={jsonList.length === 0}>Correct</button>
        </div>
      </div>
    </>
  )
}

export default App
