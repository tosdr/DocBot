import * as fs from 'fs';
import * as path from 'path';

import { Regex } from '../models';
import { Map } from '../models';
import * as color from 'chalk';

/**
 * Load all Regex' in the `regex/` folder and return a collection of
 * regex.
 */
export function loadRegex() {
	const files = fs.readdirSync(path.join(__dirname, '../regex'));
	const regexs = new Map<string, Regex>();
	const errored: string[] = [];

	files.forEach((file) => {
		fs.lstat(path.join(__dirname, '../regex/' + file), (err: any, stats: any) => {
			if (err) {
				console.log(color.red(`[LSTAT ERROR]`), color.cyan(file));
				return;
			}


			if (stats.isFile()) {

				const regex = require(`../regex/${file}`) as Regex;

				if (!regex) {
					errored.push(file);
					console.log(color.red(`Failed to load regex file`), color.cyan(file));
				} else {
					console.log(color.magenta(`Loaded regex`), color.cyan(regex.expression));
					regexs.set(regex.caseID.toString(), regex);
				}

			}

		});
	});

	if (errored.length >= 1) {
		console.log(color.yellow('A few regex\' have failed to load: %s', color.cyan(errored.join(', '))));
		console.log(color.green(`Successfully loaded ${color.cyan(regexs.size)} commands`));
	} else {
		console.log(color.green(`Successfully loaded all ${color.cyan(regexs.size)} commands!`));
	}

	return regexs;
}