import { exec } from "child_process";
import * as JSON5 from 'json5';

export function match(sentence: any, caseObj: any, document: any, service: any, quoteStart: any, quoteEnd: any, type: number) {
	let data = {
		message: "match",
		error: false,
		code: 1,
		parameters: {
			sentence: sentence,
			case: caseObj,
			document: document,
			service: service,
			quotes: {
				start: quoteStart,
				end: quoteEnd
			},
		}
	};
	switch (type) {
		case 0:
			return JSON.stringify(data);
		case 1:
			return JSON5.stringify(data);
		default:
			throw Error("Invalid Type");
	}
}