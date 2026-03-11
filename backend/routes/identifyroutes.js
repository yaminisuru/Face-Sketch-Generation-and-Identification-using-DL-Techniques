const express = require("express");
const router = express.Router();
const multer = require("multer");
const { identifyFace } = require("../controllers/identifycontroller");

const upload = multer({ dest: "uploads/" });

router.post("/identify", upload.single("sketch"), identifyFace);

module.exports = router;