import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*Google Analytics)|((?=.*analytics)|(?=.*statistics))((?=.*third)|(?=.*party)|(?=.*parties)))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 325,
	name: "This service uses third-party cookies for statistics"
} as Regex;