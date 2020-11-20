import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*Español)|(?=.*Français)|(?=.*Deutsch)|(?=.*Português)|(?=.*Italiano)|(?=.*Polski)|(?=.*language\:))", "i"),
	caseID: 198
} as Regex;