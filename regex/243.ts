import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*bankruptcy)|(?=.*bankrupt)|(?=.*merger)|(?=.*merged)|(?=.*business assets))", "i"),
	caseID: 243
} as Regex;