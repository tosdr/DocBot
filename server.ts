import { loadRegex } from './utils/load-regex';
import { isJson, isJson5 } from './utils/check-json';
import * as Response from './utils/response/index';
import * as process from 'process';
import * as express from 'express';
import * as http from 'http';
import * as striptags from 'striptags';
import * as https from 'https';
import * as WebSocket from 'ws';
import * as dotenv from 'dotenv';
import * as color from 'chalk';
import { v4 as uuidv4 } from 'uuid';
import { isPrimitive } from 'util';
import * as url from 'url';
import { exec } from 'child_process';
import * as JSON5 from 'json5';


console.log(color.green(`
    ____             ____        __ 
   / __ \____  _____/ __ )____  / /_
  / / / / __ \/ ___/ __  / __ \/ __/
 / /_/ / /_/ / /__/ /_/ / /_/ / /_  
/_____/\____/\___/_____/\____/\__/  
              Server                        

`));
console.log(color.red(`BETA Project by Justin Back`));

console.log(color.blue("********************** Initializing ************************"));

export const regexs = loadRegex();


const app = express();
dotenv.config();
console.log(color.green("Loaded environment variables!"));


// initialize a simple http server
const server = http.createServer(app);

// initialize the WebSocket server instance
const wss = new WebSocket.Server({ server });

server.listen(parseInt(process.argv[2]), '0.0.0.0', () => {
	console.log(color.green(`Gateway server started on port ${color.cyan(process.argv[2])}`));
	console.log(color.blue("********************** Done ************************"));
	console.log(color.green("Server is now ready to accept connections"));
});

wss.on('connection', async (ws: any, req: any) => {


	ws.send = (function (_super) {
		return function () {
			// Extend it to log the value for example that is passed
			if (parseInt(process.env.VERBOSITY) >= 2) {
				console.log(color.cyan("[Outgoing Message]"), color.magenta(`<${ipAddress}>`), color.yellow(arguments[0]));
			}
			return _super.apply(this, arguments);
		};

	})(ws.send);

	let ipAddress = req.socket.remoteAddress;
	if (ipAddress === null || typeof ipAddress === "undefined" || ipAddress === "127.0.0.1") {
		if (req.headers['x-forwarded-for']) {
			ipAddress = req.headers['x-forwarded-for'].split(/\s*,\s*/)[0];
		}
		console.log(req.headers['x-forwarded-for']);
	}


	ws.ipAddress = ipAddress;
	ws.sessionID = uuidv4();



	if (parseInt(process.env.VERBOSITY) >= 2) {
		console.log(color.green("[Session Created]"), color.magenta(`<${ipAddress}>`), color.green(ws.sessionID));
	}

	if (parseInt(process.env.VERBOSITY) >= 1) {
		console.log(color.green("[Connection Open]"), color.magenta(`<${ipAddress}>`), color.green(req.url));
	}
	ws.on('close', async (code: any, reason: any) => {
		//commands.get("close_connection")!.execute(ws, reason, wss);

		if (parseInt(process.env.VERBOSITY) >= 1) {
			console.log(color.yellow("[Connection Close]"), color.magenta(`<${ipAddress}>`), color.red(code), color.green(reason));
		}
	});

	ws.on('error', async (err: any) => {
		//commands.get("close_connection")!.execute(ws, err, wss);

		if (parseInt(process.env.VERBOSITY) >= 1) {
			console.log(color.red("[Connection Error]"), color.magenta(`<${ipAddress}>`), color.yellow(err));
		}
	});

	let Instance = req.url;
	let Query = url.parse(Instance, true).query;

	if (Query !== null) {
		if (("instance" in Query)) {
			Instance = Query.instance;
		} else {
			Instance = Instance.substr(1);
		}
	} else {
		Instance = Instance.substr(1);
	}


	if (("Content-Type" in Query)) {
		switch (Query["Content-Type"]) {
			case "json":
				ws.ContentType = 0;
				break;
			case "json5":
				ws.ContentType = 1;
				break;
			default:
				ws.ContentType = 1;
		}
	}



	ws.on('message', async (message: string) => {
		if (parseInt(process.env.VERBOSITY) >= 2) {
			console.log(color.cyan("[Incoming Message]"), color.magenta(`<${ipAddress}>`), color.yellow(message));
		}



		if (ws.ContentType === 0) {
			if (!isJson(message)) {
				ws.send(Response.error("invalid_json", 0, ws.ContentType));
				return false;
			}
		} else if (ws.ContentType === 1) {
			if (!isJson5(message)) {
				ws.send(Response.error("invalid_json5", 0, ws.ContentType));
				return false;
			}
		} else {
			ws.send(Response.error("invalid_content_type", 0));
			return false;
		}

		try {

			let messageJSON;
			if (ws.ContentType === 0) {
				messageJSON = JSON.parse(message);
			} else if (ws.ContentType === 1) {
				messageJSON = JSON5.parse(message);
			} else {
				ws.send(Response.error("invalid_content_type", 0));
				return false;
			}

			try {

				if (!("api_key" in messageJSON)) {
					ws.send(Response.error("missing_parameter_api", 3, ws.ContentType));
					return;
				}
				if (process.env.API_KEY !== messageJSON.api_key) {
					ws.send(Response.error("api_key_mismatch", 0, ws.ContentType));
					return;
				}

			} catch (error) {
				ws.send(Response.error("error_exec", 0, ws.ContentType));
				if (parseInt(process.env.VERBOSITY) >= 3) {
					console.log(color.red("[ERROR]", color.magenta("<JSON Exception Stacktrace>"), color.yellow(error.stack)));
				} else {
					console.log(color.red("[ERROR]", color.magenta("<JSON Exception>"), color.yellow(error)));
				}
			}
			try {
				regexs.forEach((RegularExpression) => {

					const options = {
						hostname: 'edit.tosdr.org',
						port: 443,
						path: '/api/v1/services/' + messageJSON.service,
						method: 'GET'
					}
					let str = "";
					let matches = [];

					const req = https.request(options, response => {
						console.log("Sending Request");
						response.on('data', function (chunk) {
							console.log("Parsing Phoenix");
							str += chunk;
						});

						//the whole response has been received, so we just print it out here
						response.on('end', () => {
							console.log("Phoenix parsed");
							let parsed = JSON.parse(str);
							for (var documentIndex in parsed.documents) {
								console.log("Found document", parsed.documents[documentIndex].name);
								if (parsed.documents[documentIndex].text === null) {
									continue;
								}
								let Sentences = parsed.documents[documentIndex].text.split(".\n");
								for (var index in Sentences) {
									if (RegularExpression.expression.test(Sentences[index])) {

										let quoteStart = parsed.documents[documentIndex].text.indexOf(Sentences[index]);
										let quoteEnd = quoteStart + Sentences[index].length;

										ws.send(Response.match(striptags(Sentences[index]).replace(/\n/g, ''), RegularExpression.caseID, parsed.documents[documentIndex].id, quoteStart, quoteEnd, ws.ContentType))
									}
								}
							}
						});
					})

					req.on('error', error => {
						console.error(error)
					})

					req.end();

				});
			} catch (error) {
				ws.send(Response.error("error_exec", 0, ws.ContentType));
				console.log(color.red("[ERROR]", color.magenta(`<${messageJSON.service}>`), color.yellow(error)));
			}
		} catch (error) {
			ws.send(Response.error("error_exec", 0, ws.ContentType));
			if (parseInt(process.env.VERBOSITY) >= 3) {
				console.log(color.red("[ERROR]", color.magenta("<Global Exception Stacktrace>"), color.yellow(error.stack)));
			} else {
				console.log(color.red("[ERROR]", color.magenta("<Global Exception>"), color.yellow(error)));
			}
		}
	});

});