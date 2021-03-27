import * as fs from 'fs';
import * as path from 'path';

const pkg = require('../package');
import { Case } from '../models';
import { Map } from '../models';
import * as color from 'chalk';
import * as request from 'request';

/**
 * Load all Regex' in the `regex/` folder and return a collection of
 * regex.
 */
export function loadRegex() {
    const regexs = new Map<string, Case>();
    const errored: string[] = [];

    const options = {
        url: 'https://api.tosdr.org/all-cases/v1/nocache',
        headers: {
            'User-Agent': 'DocBotServer/' + pkg.version + ' (+https://github.com/tosdr/DocBot-Server)'
        }
    };

    request(options, function (error, response, body) {
        if (error) {
            console.log(color.red(`[REQUEST ERROR]`), color.cyan(error));
            return;
        }

        body = JSON.parse(body);


        for (var caseInfo of body.parameters) {


            let caseObj = caseInfo as Case;

            if (caseObj.docbot_regex == null || caseObj.docbot_regex == "") {
                continue;
            }

            console.log("Loading", caseObj.id, caseObj.title);

            const regex = new RegExp(caseInfo.docbot_regex, "mi");

            if (!regex) {
                errored.push(caseInfo.id);
                console.log(color.red(`Failed to load regex`), color.cyan(caseObj.id));
            } else {
                caseObj.compiled_regex = regex;
                console.log(color.magenta(`Loaded regex`), color.cyan(caseObj.docbot_regex), color.red(caseObj.id));
                regexs.set(caseObj.id.toString(), caseObj);
            }

            if (errored.length >= 1 && body.parameters.length - 1 == regexs.size) {
                console.log(color.yellow('A few regex\' have failed to load: %s', color.cyan(errored.join(', '))));
            } else if (body.parameters.length - 1 == regexs.size) {
                console.log(color.green(`Successfully loaded all ${color.cyan(regexs.size)} regex files!`));
            }
        }



    });


    return regexs;
}