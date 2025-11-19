import React from "react";
import { Routes, Route } from "react-router-dom";
import SimilarPeople from "./pages/SimilarPeople";
import Home from "./pages/Home.js";

function App() {
  

  return (
    <>
    <Routes>
      <Route path="/similar" element={<SimilarPeople />} />
      <Route path="/" element={<Home />} />
    </Routes>
    </>
  );
}

export default App;
