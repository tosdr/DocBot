import * as JSON5 from 'json5';

export function isJson(json: any) {
	try {
		JSON.parse(json);
	} catch (e) {
		return false;
	}
	return true;
}
export function isJson5(json: any) {
	try {
		JSON5.parse(json);
	} catch (e) {
		return false;
	}
	return true;
}