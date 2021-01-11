import { Regex } from '../models';

module.exports = {
    expression: new RegExp("^((?=.*indemnify)(harmless))", "mi"),
	expressionDont: new RegExp("^((?=.*defend))", "i"),
	caseID: 145,
	name: "You are solely responsible for claims made against the service and agree to indemnify and hold harmless the service."
} as Regex;