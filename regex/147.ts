import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*survive))((?=.*termination))", "i"),
	caseID: 147,
	name: "defend, indemnify, hold harmless; survives termination"
} as Regex;