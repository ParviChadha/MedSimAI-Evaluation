const fs = require('fs');

// Read the dialogs JSON file
function convertDialogsToTranscript(inputFile, outputFile) {
  // Read the JSON file
  fs.readFile(inputFile, 'utf8', (err, data) => {
    if (err) {
      console.error(`Error reading file: ${err}`);
      return;
    }

    try {
      // Parse the JSON data
      const dialogsData = JSON.parse(data);
      let transcript = '';

      // Process each dialog
      Object.values(dialogsData).forEach(dialog => {
        // Process each utterance in the dialog
        dialog.utterances.forEach(utterance => {
          // Format the speaker role properly
          let speakerRole;
          if (utterance.speaker === "doctor") {
            speakerRole = "Medical Student";
          } else {
            // Capitalize first letter for other roles
            speakerRole = utterance.speaker.charAt(0).toUpperCase() + utterance.speaker.slice(1);
          }
          
          // Format the line according to the specified format
          transcript += `**${speakerRole}: **${utterance.text}\n`;
        });
        
        // Add a newline between different dialogs
        transcript += '\n';
      });

      // Write the transcript to the output file
      fs.writeFile(outputFile, transcript, err => {
        if (err) {
          console.error(`Error writing transcript file: ${err}`);
          return;
        }
        console.log(`Transcript successfully written to ${outputFile}`);
      });
    } catch (error) {
      console.error(`Error parsing JSON: ${error}`);
    }
  });
}

// Usage example
const inputFile = 'dialogs110.json';
const outputFile = 'transcript110.txt';
convertDialogsToTranscript(inputFile, outputFile);