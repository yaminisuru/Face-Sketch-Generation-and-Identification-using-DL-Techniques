import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/signin.css";
import axios from "axios";

function SignIn(){

  const navigate = useNavigate();

  const [username,setUsername] = useState("");
  const [password,setPassword] = useState("");
  const [error,setError] = useState("");

  const handleLogin = async (e)=>{
    e.preventDefault();

    try{

      const res = await axios.post(
        "http://localhost:5000/api/auth/login",
        {
          username: username,
          password: password
        }
      );

      localStorage.setItem("loggedInUser",username);

      navigate("/options");

    }catch(error){

      if(error.response){
        setError(error.response.data);
      }else{
        setError("Server error");
      }

    }
  };

  return(

    <div className="auth-container">

      <div className="auth-card">

        <h2>Face Sketch System</h2>

        <p className="project-subtitle">
          Criminal Identification Platform
        </p>

        <form onSubmit={handleLogin}>

          <input
            type="text"
            placeholder="Username"
            value={username}
            onChange={(e)=>setUsername(e.target.value)}
            required
          />

          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e)=>setPassword(e.target.value)}
            required
          />

          {error && <p className="error">{error}</p>}

          <button type="submit">
            Sign In
          </button>

        </form>

        <p className="auth-link">
          New user?{" "}
          <span onClick={()=>navigate("/register")}>
            Register
          </span>
        </p>

      </div>

    </div>
  );
}

export default SignIn;