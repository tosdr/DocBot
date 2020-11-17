import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*complaint))"),
	caseID: 300
} as Regex;