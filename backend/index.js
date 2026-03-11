const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const authRoute = require("./routes/authroutes");
const identifyRoute = require("./routes/identifyroutes");

const app = express();

app.use(cors());
app.use(express.json());

// serve images
app.use("/photos", express.static("dl_model/processed_data/Photos"));

mongoose.connect("mongodb://127.0.0.1:27017/facesketch")
  .then(() => console.log("MongoDB Connected"))
  .catch(err => console.log("MongoDB connection error:", err));

app.use("/api/auth", authRoute);
app.use("/api", identifyRoute);

const PORT = 5000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});