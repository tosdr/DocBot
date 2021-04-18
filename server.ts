import { loadRegex } from './utils/load-regex';
import { isJson, isJson5 } from './utils/check-json';
import * as Response from './utils/response/index';
import * as process from 'process';
import * as express from 'express';
import * as http from 'http';
import * as fs from 'fs';
import * as striptags from 'striptags';
import * as https from 'https';
import * as WebSocket from 'ws';
import * as dotenv from 'dotenv';
import * as color from 'chalk';
import { v4 as uuidv4 } from 'uuid';
import * as url from 'url';
import * as JSON5 from 'json5';
import * as Package from './package.json';
import * as crypto from 'crypto';
import * as publicIp from 'public-ip';
import { exit } from 'process';
function objToString(obj) {
    var str = '';
    for (var p in obj) {
        if (obj.hasOwnProperty(p)) {
            str += p + '=' + obj[p] + '\n';
        }
    }
    return str;
}

console.log(color.green(`
    ____             ____        __ 
   / __ \____  _____/ __ )____  / /_
  / / / / __ \/ ___/ __  / __ \/ __/
 / /_/ / /_/ / /__/ /_/ / /_/ / /_  
/_____/\____/\___/_____/\____/\__/  
         Server ${Package.version}                        

`));
console.log(color.red(`BETA Project by Justin Back`));

console.log(color.blue("********************** Initializing ************************"));

export let regexs = loadRegex();



publicIp.v4().then((ip) => {



    const httpserver = http.createServer((req, res) => {
        res.writeHead(200, { 'content-type': 'text/html' })
        fs.createReadStream('docs/index.html').pipe(res)
    })



    const app = express();
    let dotenvparsed = dotenv.config();
    console.log(color.green("Loaded environment variables!"));
    dotenvparsed.parsed.API_KEY = "***";
    console.log(color.cyan(objToString(dotenvparsed.parsed)));


    const server = http.createServer(app);

    // initialize the WebSocket server instance
    const wss = new WebSocket.Server({ server });
    console.log(color.blue("************ Starting Gateway Server ***************"));

    server.listen(parseInt(process.env.GATEWAY_PORT), process.env.GATEWAY_BIND, () => {
        console.log(color.green(`Gateway server started on port ${color.cyan(process.env.GATEWAY_BIND)}:${color.cyan(process.env.GATEWAY_PORT)}`));
        console.log(color.blue("********************** Done ************************"));
        console.log(color.green("Gateway Server is now ready to accept connections"));
    });

    console.log(color.blue("************** Starting HTTP Server ****************"));
    httpserver.listen(parseInt(process.env.HTTP_PORT), process.env.HTTP_BIND, () => {
        console.log(color.green(`HTTP Web server started on port ${color.cyan(process.env.HTTP_BIND)}:${color.cyan(process.env.HTTP_PORT)}`));
        console.log(color.blue("********************** Done ************************"));
        console.log(color.green("HTTP Server is now ready to accept connections"));
    });


    console.log(color.red("********************** INFOS ************************"));
    if (ip == process.env.HTTP_BIND || process.env.HTTP_BIND == "0.0.0.0") {
        console.log(color.green(`The webserver is reachable under http://${ip}:${process.env.HTTP_PORT}`));
    } else {
        console.log(color.green(`The webserver is reachable under http://${process.env.HTTP_BIND}:${process.env.HTTP_PORT}`));
    }
    if (ip == process.env.GATEWAY_BIND || process.env.GATEWAY_BIND == "0.0.0.0") {
        console.log(color.green(`The gateway server is reachable under ws://${ip}:${process.env.GATEWAY_PORT}`));
    } else {
        console.log(color.green(`The gateway server is reachable under ws://${process.env.GATEWAY_BIND}:${process.env.GATEWAY_PORT}`));
    }
    console.log(color.blue("********************** Done ************************"));

    wss.on('connection', async (ws: any, req: any) => {


        ws.send = (function (_super) {
            return function () {
                // Extend it to log the value for example that is passed
                if (parseInt(process.env.VERBOSITY) >= 2) {
                    console.log(color.cyan("[Outgoing Message]"), color.magenta(`<${ws.sessionID}>`), color.yellow(arguments[0]));
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
            console.log(color.green("[Session Created]"), color.magenta(`<${ws.sessionID}>`), color.green(ws.sessionID));
        }

        if (parseInt(process.env.VERBOSITY) >= 1) {
            console.log(color.green("[Connection Open]"), color.magenta(`<${ws.sessionID}>`), color.green(req.url));
        }
        ws.on('close', async (code: any, reason: any) => {
            //commands.get("close_connection")!.execute(ws, reason, wss);

            if (parseInt(process.env.VERBOSITY) >= 1) {
                console.log(color.yellow("[Connection Close]"), color.magenta(`<${ws.sessionID}>`), color.red(code), color.green(reason));
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



        if (("Content-Type" in Query)) {
            switch (Query["Content-Type"]) {
                case "json":
                    ws.ContentType = 0;
                    break;
                case "json5":
                    ws.ContentType = 1;
                    break;
                default:
                    ws.ContentType = 0;
            }
        } else {
            ws.ContentType = 0;
        }

        ws.send(Response.message("session", { sessionID: ws.sessionID }, ws.ContentType));



        ws.on('message', async (message: string) => {
            if (parseInt(process.env.VERBOSITY) >= 2) {
                console.log(color.cyan("[Incoming Message]"), color.magenta(`<${ws.sessionID}>`), color.yellow(message));
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


                } catch (error) {
                    ws.send(Response.error("error_exec", 0, ws.ContentType));
                    if (parseInt(process.env.VERBOSITY) >= 3) {
                        console.log(color.red("[ERROR]", color.magenta("<JSON Exception Stacktrace>"), color.yellow(error.stack)));
                    } else {
                        console.log(color.red("[ERROR]", color.magenta("<JSON Exception>"), color.yellow(error)));
                    }
                }
                try {




                    let strkey = "";
                    const keyreq = https.request({
                        hostname: 'api.tosdr.org',
                        port: 443,
                        path: '/apikey/v1/?apikey=' + messageJSON.api_key,
                        method: 'GET',
                        headers: {
                            'User-Agent': 'DocBotServer/' + Package.version + ' (+https://github.com/tosdr/DocBot-Server)',
                            'Authorization': process.env.API_KEY
                        }
                    }, keyresponse => {
                        keyresponse.on('data', function (chunk) {
                            if (parseInt(process.env.VERBOSITY) >= 3) {
                                console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.green("Received API Key Chunk"), color.red(crypto.createHash('md5').update(chunk).digest("hex")));
                            }
                            strkey += chunk;
                        });


                        keyresponse.on('end', () => {

                            if (keyresponse.statusCode === 403 || keyresponse.statusCode === 401) {
                                ws.send(Response.error("server_error", -1, ws.ContentType));
                                console.log("Invalid Server API Key. Misconfigured");
                                return;
                            } else if (keyresponse.statusCode !== 200) {
                                ws.send(Response.error("server_error", -1, ws.ContentType));
                                console.log("API Key request failed", keyresponse.statusMessage);
                                return;
                            }

                            let apiresponse = JSON.parse(strkey);

                            console.log(apiresponse);


                            if (!(apiresponse.error & 0x100) || !(apiresponse.parameters.permissions & 0x10) || apiresponse.parameters.revoked || (apiresponse.parameters.expires_at !== null || Math.floor(new Date(apiresponse.parameters.expires_at).getTime()) > Math.floor(new Date().getTime()))) {
                                ws.send(Response.error("api_key_mismatch", 0, ws.ContentType));
                                return;
                            }

                            if (apiresponse.parameters.expires_at !== null || Math.floor(new Date(apiresponse.parameters.expires_at).getTime()) < Math.floor(Date.now())) {
                                ws.send(Response.error("api_key_expired", 0, ws.ContentType));
                                return;
                            }

                            if (("reload" in messageJSON)) {
                                regexs = loadRegex();
                                ws.send(Response.message("reloaded", null, ws.ContentType));
                                return;
                            }

                            if (!("service" in messageJSON)) {
                                ws.send(Response.error("missing_parameter_service", 3, ws.ContentType));
                                return;
                            }



                            console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.green("Received Crawl Request for Service"), color.red(messageJSON.service));


                            const options = {
                                hostname: 'api.tosdr.org',
                                port: 443,
                                path: '/rest-service/v3/' + messageJSON.service,
                                method: 'GET',
                                headers: {
                                    'User-Agent': 'DocBotServer/' + Package.version + ' (+https://github.com/tosdr/DocBot-Server)',
                                    'Authorization': process.env.API_KEY
                                }
                            }
                            let str = "";
                            let matches = [];
                            console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.green("Starting Request to Service"), color.magenta(messageJSON.service), color.red(crypto.createHash('md5').update(messageJSON.service).digest("hex")));

                            const req = https.request(options, response => {
                                response.on('data', function (chunk) {
                                    if (parseInt(process.env.VERBOSITY) >= 3) {
                                        console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.green("Received Phoenix Chunk"), color.red(crypto.createHash('md5').update(chunk).digest("hex")));
                                    }
                                    str += chunk;
                                });

                                response.on('end', () => {
                                    console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.cyan("Digested Phoenix Chunk(s)"), color.red(crypto.createHash('md5').update(str).digest("hex")));
                                    let parsed = JSON.parse(str).parameters;
                                    let count = 0;
                                    for (var documentIndex in parsed.documents) {

                                        //count++;

                                        if (parsed.documents[documentIndex].text === null || parsed.documents[documentIndex].text === "") {
                                            console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.yellow("Skipping document, it's empty!"));
                                            continue;
                                        }

                                        console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.green("Parsed Document"), color.green(parsed.documents[documentIndex].name), color.red(crypto.createHash('md5').update(parsed.documents[documentIndex].text).digest("hex")));

                                        let Sentences = parsed.documents[documentIndex].text.split(".\n");
                                        console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.cyan("Parsing Sentences"), color.red(crypto.createHash('md5').update(parsed.documents[documentIndex].text).digest("hex")));



                                        regexs.forEach((RegularExpression) => {

                                            for (var index in Sentences) {
                                                if (RegularExpression.compiled_regex.test(Sentences[index]) && (!matches.includes(RegularExpression.id) && process.env.NO_DUPLICATES == "1")) {


                                                    let quoteStart = parsed.documents[documentIndex].text.indexOf(Sentences[index]);
                                                    let quoteEnd = quoteStart + Sentences[index].length;

                                                    console.log(color.magenta("[DocBot]"), color.magenta(`<${ws.sessionID}>`), color.green("I have found a match on Line"), color.magenta(index), color.green("for the case"), color.magenta(RegularExpression.id), color.red(crypto.createHash('md5').update(Sentences[index]).digest("hex")));

                                                    ws.send(Response.match(
                                                        striptags(Sentences[index]).replace(/\n/g, ''),
                                                        RegularExpression,
                                                        parsed.documents[documentIndex],
                                                        {
                                                            name: parsed.name,
                                                            url: parsed.url,
                                                            rating: parsed.rating
                                                        },
                                                        quoteStart,
                                                        quoteEnd,
                                                        ws.ContentType
                                                    ));
                                                    matches.push(RegularExpression.id);



                                                }

                                            }

                                        });
                                    }
                                    ws.send(Response.message("finished", null, ws.ContentType));
                                });
                            })

                            req.on('error', error => {
                                console.error(error)
                            })

                            req.end();
                        });
                    })

                    keyreq.on('error', error => {
                        console.error(error)
                    })

                    keyreq.end();
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
})