import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*complaint))", "i"),
	caseID: 300
} as Regex;