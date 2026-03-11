import React, { useState } from "react";
import { DndProvider } from "react-dnd";
import { HTML5Backend } from "react-dnd-html5-backend";
import "../styles/sketchapp.css";
import Sidebar from "../components/Sidebar";
import CanvasArea from "../components/FaceCanvas";
import RightPanel from "../components/RightPanel";

function App() {

  const [selectedPart, setSelectedPart] = useState(null);
  const [elements, setElements] = useState({});
  const [history, setHistory] = useState([]);

  return (
    <DndProvider backend={HTML5Backend}>
      <div className="app-container">

        <Sidebar setSelectedPart={setSelectedPart} />

        <CanvasArea
          elements={elements}
          setElements={setElements}
          history={history}
          setHistory={setHistory}
          selectedPart={selectedPart}
          setSelectedPart={setSelectedPart}
        />

        <RightPanel
          selectedPart={selectedPart}
          setSelectedPart={setSelectedPart}
          elements={elements}
          setElements={setElements}
          history={history}
          setHistory={setHistory}
        />

      </div>
    </DndProvider>
  );
}

export default App;