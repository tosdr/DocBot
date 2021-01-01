import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*bankruptcy)|(?=.*bankrupt)|(?=.*merger)|(?=.*merged)|(?=.*business assets))", "i"),
	caseID: 243,
	name: "The service can sell or otherwise transfer your personal data as part of a bankruptcy proceeding or other type of financial transaction."
} as Regex;