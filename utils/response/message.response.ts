import { exec } from "child_process";
import * as JSON5 from 'json5';

export function message(message: any, parameter: any, type: number) {
	let data = {
		message: message,
		error: false,
		code: 1,
		parameter: parameter
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