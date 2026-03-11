import { Routes, Route } from "react-router-dom";
import SignIn from "./pages/signin";
import Register from "./pages/register";
import Options from "./pages/options";
import SketchApp from "./pages/sketchapp";
import UploadSketch from "./pages/uploadsketch";

function App(){

  return(

    <Routes>

      <Route path="/" element={<SignIn/>}/>
      <Route path="/register" element={<Register/>}/>
      <Route path="/options" element={<Options/>}/>
      <Route path="/create" element={<SketchApp/>}/>
       <Route path="/upload" element={<UploadSketch />} />
    </Routes>

  );
}

export default App;