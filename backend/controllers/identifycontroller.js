const { exec } = require("child_process");
const path = require("path");

exports.identifyFace = (req, res) => {

  const sketchPath = req.file.path;

  const scriptPath = path.join(
    __dirname,
    "../dl_model/identify.py"
  );

  exec(`python "${scriptPath}" "${sketchPath}"`, (error, stdout, stderr) => {

    if (error) {
      console.error(error);
      return res.status(500).json({ error: "Identification failed" });
    }

    try {

      const results = JSON.parse(stdout);

      res.json({ matches: results });

    } catch (e) {

      console.error("JSON parse error:", e);
      res.status(500).json({ error: "Invalid Python output" });

    }

  });

};