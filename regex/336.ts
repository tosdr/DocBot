import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*personal)|(?=.*user))((?=.*marketing)|(?=.*information))", "i"),
	caseID: 336
} as Regex;