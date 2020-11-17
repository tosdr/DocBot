import * as JSON5 from 'json5';
export function error(message: any, errorcode: any, type: number = 0) {

	switch (type) {
		case 0:
			return JSON.stringify({ "type": "error", "result": { "error": message, "code": errorcode } });
		case 1:
			return JSON5.stringify({ "type": "error", "result": { "error": message, "code": errorcode } });
		default:
			throw Error("Invalid Type");
	}
}