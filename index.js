var myArgs = process.argv.slice(2);
let str = "";
const https = require('https')
const options = {
  hostname: 'edit.tosdr.org',
  port: 443,
  path: '/api/v1/services/'+ myArgs[0],
  method: 'GET'
}


let regex = {
	"^((?=.*dnt)|(?=.*do not track))((?=.*respond)|(?=.*recognize))": 233,
	"^((?=.*dnt)|(?=.*do not track))((?=.*respect))": 383,
	"^((?=.*bankruptcy)|(?=.*bankrupt)|(?=.*merger)|(?=.*merged)|(?=.*business assets))": 243,
	"^((?=.*throttle)|(?=.*reduce))((?=.*speed)|(?=.*bandwidth))": 157,
	"^((?=.*spidering)|(?=.*spider)|(?=.*crawler)|(?=.*crawling))((?=.*not)|(?=.*no))((?=.*automatic)|(?=.*automation))": 150,
	"^((?=.*survive))((?=.*termination))": 147,
	"^((?=.*as long as)|(?=.*purposes))((?=.*necessary)|(?=.*legally obligated))": 178,
	"^((?=.*complaint))": 300,
}

const req = https.request(options, response => {
  response.on('data', function (chunk) {
    str += chunk;
  });

  //the whole response has been received, so we just print it out here
  response.on('end', function () {
    str = JSON.parse(str);
	for(var documentIndex in str.documents){
		if(str.documents[documentIndex].text === null){
			continue;
		}
		let Sentences = str.documents[documentIndex].text.split(".\n");
		for(var index in Sentences){
			for(let [key, value] of Object.entries(regex)){
				key = new RegExp(key);
				if(key.test(Sentences[index])){
					console.log("Found a match in the", str.documents[documentIndex].name, str.documents[documentIndex].id);
					console.log("Sentence:", Sentences[index], "| CaseID:", value, "\n");
				}
			}
		}
	}
  });
})

req.on('error', error => {
  console.error(error)
})

req.end()