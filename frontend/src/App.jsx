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
  const [useFormInput, setUseFormInput] = useState(false);
  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

  const handleLoadModel = async () => {
    const response = await fetch(`${apiBaseUrl}/load_model`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model_type: modelType }),
    });
    setJsonList([]);
    const data = await response.json();
    setOutputText(JSON.stringify(data, null, 2));
  };

  const handleGenerate = async () => {
    const response = await fetch(`${apiBaseUrl}/generate`, {
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
      const response = await fetch(`${apiBaseUrl}/correct`, {
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

  const handleToggleChange = (checked) => {
    if (!checked) { // Switching to JSON mode
      setUseFormInput(false);
    } else { // Switching to Form mode
      try {
        // Parse the current JSON editor content and update the state
        const updatedJson = JSON.parse(outputText);
        const updatedJsonList = [...jsonList];
        updatedJsonList[currentIndex] = updatedJson;
        setJsonList(updatedJsonList);
        setUseFormInput(true);
      } catch (error) {
        alert('Invalid JSON format. Please correct before switching to form mode.');
        return;
      }
    }
  };

  const renderFormInput = () => {
    if (!jsonList.length || !jsonList[currentIndex]?.target_json) return null;

    const currentItem = jsonList[currentIndex];
    const targetJson = currentItem.target_json;

    const handleFieldChange = (index, newValue) => {
      const updatedJsonList = [...jsonList];
      updatedJsonList[currentIndex].target_json[index].value = newValue;
      setJsonList(updatedJsonList);
      setOutputText(JSON.stringify(updatedJsonList[currentIndex], null, 2));
    };

    return (
      <div className="form-container">
        {targetJson.map((item, index) => (
          <div key={index} className={`form-item ${item.type === 'boolean' ? 'boolean-input' : ''}`}>
            <label>{item.field}</label>
            {item.type === 'enum' ? (
              <select
                value={item.value || ''}
                onChange={(e) => handleFieldChange(index, e.target.value)}
              >
                <option value="">Select an option</option>
                {item.enum.map((option, optIndex) => (
                  <option key={optIndex} value={option.value}>
                    {option.name || option.value}
                    {option.group ? ` (${option.group})` : ''}
                  </option>
                ))}
              </select>
            ) : item.type === 'int' ? (
              <input
                type="number"
                value={item.value || ''}
                onChange={(e) => handleFieldChange(index, e.target.value ? parseInt(e.target.value) : null)}
                placeholder="Enter a number"
              />
            ) : item.type === 'boolean' ? (
              <input
                type="checkbox"
                checked={item.value || false}
                onChange={(e) => handleFieldChange(index, e.target.checked)}
              />
            ) : (
              <input
                type="text"
                value={item.value || ''}
                onChange={(e) => handleFieldChange(index, e.target.value)}
                placeholder="Enter text"
              />
            )}
          </div>
        ))}
      </div>
    );
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
        <button onClick={handleLoadModel} className="action-button">1. Load Model</button>
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
          <button onClick={handleGenerate} className="action-button">2. Generate</button>
        </div>
        <div className="right-panel">
          <div className="editor-toggle">
            <h4>Form Input: </h4>
            <label className="switch">
              <input
                type="checkbox"
                checked={useFormInput}
                onChange={(e) => handleToggleChange(e.target.checked)}
              />
              <span className="slider round"></span>
            </label>
          </div>
          <div className="editor-container">
            {useFormInput ? (
              renderFormInput()
            ) : (
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
                  automaticLayout: true,
                  lineNumbers: 'on',
                  lineNumbersMinChars: 3
                }}
              />
            )}
          </div>
          <div className="navigation-controls">
            <button onClick={handlePrevious} disabled={currentIndex === 0 || jsonList.length === 0}>Prev</button>
            <span>{jsonList.length > 0 ? `${currentIndex + 1} / ${jsonList.length}` : 'No data'}</span>
            <button onClick={handleNext} disabled={currentIndex === jsonList.length - 1 || jsonList.length === 0}>Next</button>
          </div>
          <button onClick={handleCorrect} className="action-button" disabled={jsonList.length === 0}>3. Submit Correction</button>
        </div>
      </div>
    </>
  )
}

export default App
