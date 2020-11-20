import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*unauthorized)(?=.*computer))", "i"),
	caseID: 284
} as Regex;