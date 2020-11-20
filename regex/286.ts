import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*\"as is\")|(?=.*\"as available\"))", "i"),
	caseID: 286
} as Regex;