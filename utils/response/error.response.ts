import * as JSON5 from 'json5';
export function error(message: any, errorcode: any, type: number = 0) {

	let data = {
		message: "error",
		error: message,
		code: errorcode,
		parameters: {}
	};
	switch (type) {
		case 0:
			return JSON.stringify({ "message": "error", "parameters": { "error": message, "code": errorcode } });
		case 1:
			return JSON5.stringify({ "message": "error", "parameters": { "error": message, "code": errorcode } });
		default:
			throw Error("Invalid Type");
	}
}