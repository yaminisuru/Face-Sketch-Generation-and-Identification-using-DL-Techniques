const router = require("express").Router();
const User = require("../models/user");
const bcrypt = require("bcryptjs");

/* REGISTER */

router.post("/register", async(req,res)=>{

  try{

    const hashedPassword = await bcrypt.hash(req.body.password,10);

    const newUser = new User({
      username:req.body.username,
      password:hashedPassword
    });

    await newUser.save();

    res.json("User registered successfully");

  }catch(err){

    res.status(500).json(err);

  }

});


/* LOGIN */

router.post("/login", async(req,res)=>{

  try{

    const user = await User.findOne({
      username:req.body.username
    });

    if(!user){
      return res.status(400).json("User not found");
    }

    const validPassword = await bcrypt.compare(
      req.body.password,
      user.password
    );

    if(!validPassword){
      return res.status(400).json("Invalid password");
    }

    res.json("Login successful");

  }catch(err){

    res.status(500).json(err);

  }

});

module.exports = router;