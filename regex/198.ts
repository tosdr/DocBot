import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*Español)|(?=.*Français)|(?=.*Deutsch)|(?=.*Português)|(?=.*Italiano)|(?=.*Polski)|(?=.*language\:))", "i"),
	caseID: 198,
	name: "The terms for this service are translated into different languages"
} as Regex;