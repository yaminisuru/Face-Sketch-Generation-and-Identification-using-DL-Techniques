import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "../styles/signin.css";
import axios from "axios";

function Register(){

  const navigate = useNavigate();

  const [username,setUsername] = useState("");
  const [password,setPassword] = useState("");
  const [message,setMessage] = useState("");

  const handleRegister = async (e)=>{
    e.preventDefault();

    try{

      const res = await axios.post(
        "http://localhost:5000/api/auth/register",
        {
          username: username,
          password: password
        }
      );

      setMessage("Registration successful");

      setTimeout(()=>{
        navigate("/");
      },1500);

    }catch(error){

      if(error.response){
        setMessage(error.response.data);
      }else{
        setMessage("Server error");
      }

    }
  };

  return(

    <div className="auth-container">

      <div className="auth-card">

        <h2>Create Account</h2>

        <p className="project-subtitle">
          Face Sketch Identification System
        </p>

        <form onSubmit={handleRegister}>

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

          {message && <p className="error">{message}</p>}

          <button type="submit">
            Register
          </button>

        </form>

        <p className="auth-link">
          Already have account?{" "}
          <span onClick={()=>navigate("/")}>
            Sign In
          </span>
        </p>

      </div>

    </div>
  );
}

export default Register;