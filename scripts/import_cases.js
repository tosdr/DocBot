const https = require('https');


const options = {
	hostname: 'edit.tosdr.org',
	port: 443,
	path: '/api/v1/cases/',
	method: 'GET'
}
let str = "";
const req = https.request(options, response => {
	response.on('data', function (chunk) {
		str += chunk;
	});
	//the whole response has been received, so we just print it out here
	response.on('end', () => {
		let json = JSON.parse(str);



	});
});

req.on('error', error => {
	console.error(error)
})

req.end();