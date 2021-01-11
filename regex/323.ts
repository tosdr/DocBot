import { Regex } from '../models';

module.exports = {
	expression: new RegExp("^((?=.*web beacons)|(?=.*pixel tags)|(?=.*tracking pixels)|(?=.* fingerprinting))", "mi"),
	expressionDont: new RegExp("", "i"),
	caseID: 323,
	name: "The service may use tracking pixels, web beacons, browser fingerprinting, and/or device fingerprinting on users."
} as Regex;